// based on https://github.com/linux-rdma/rdma-core/blob/master/librdmacm/examples/rping.c
// and NVIDIA's "RDMA Aware Networks Programming User Manual"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <endian.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>

#define CHUNK (8u << 20)  // bytes per RDMA READ

// rping.c defines these too: the wire format below is big-endian.
#define htonll(x) htobe64(x)
#define ntohll(x) be64toh(x)

// What the server SENDs and the client RECVs, as rping.c's rping_rdma_info.
// 64-bit length, since the datasets exceed rping's uint32_t.
struct rdma_info {
  uint64_t addr;
  uint64_t len;
  uint32_t rkey;
};

static __thread char g_err[256];

// Record a message and bail out to the function's cleanup label.
#define FAIL(...)                                \
  do {                                           \
    snprintf(g_err, sizeof(g_err), __VA_ARGS__); \
    rc = -1;                                     \
    goto out;                                    \
  } while (0)

#define FAIL_ERRNO(call) FAIL("%s: %s", call, strerror(errno))

const char *rdma_errmsg(void) { return g_err; }

static double now(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

struct ctx {
  struct rdma_event_channel *ec;
  struct rdma_cm_id *listen_id;
  struct rdma_cm_id *id;  // the connected id
  struct ibv_pd *pd;
  struct ibv_comp_channel *comp_ch;
  struct ibv_cq *cq;
  struct ibv_mr *data_mr;  // the dataset (server) or the sink (client)
  struct ibv_mr *info_mr;  // the small SEND/RECV buffer
  struct rdma_info info;
};

// Block until the next CM event, which must be the expected one, and ack it.
static int expect_cm_event(struct ctx *c, enum rdma_cm_event_type want,
                           struct rdma_cm_id **out_id) {
  struct rdma_cm_event *ev;
  if (rdma_get_cm_event(c->ec, &ev)) {
    snprintf(g_err, sizeof(g_err), "rdma_get_cm_event: %s", strerror(errno));
    return -1;
  }
  int bad = ev->event != want;
  if (bad)
    snprintf(g_err, sizeof(g_err), "expected %s, got %s", rdma_event_str(want),
             rdma_event_str(ev->event));
  else if (out_id)
    *out_id = ev->id;
  rdma_ack_cm_event(ev);
  return bad ? -1 : 0;
}

// PD, completion channel, CQ and QP, in the order the manual uses.
static int build_connection(struct ctx *c) {
  int rc = 0;
  c->pd = ibv_alloc_pd(c->id->verbs);
  if (!c->pd) FAIL_ERRNO("ibv_alloc_pd");
  c->comp_ch = ibv_create_comp_channel(c->id->verbs);
  if (!c->comp_ch) FAIL_ERRNO("ibv_create_comp_channel");
  c->cq = ibv_create_cq(c->id->verbs, 16, NULL, c->comp_ch, 0);
  if (!c->cq) FAIL_ERRNO("ibv_create_cq");
  // arm before anything is posted, so no completion can be missed
  if (ibv_req_notify_cq(c->cq, 0)) FAIL_ERRNO("ibv_req_notify_cq");

  struct ibv_qp_init_attr attr = {0};
  attr.send_cq = c->cq;
  attr.recv_cq = c->cq;
  attr.qp_type = IBV_QPT_RC;
  attr.cap.max_send_wr = 4;
  attr.cap.max_recv_wr = 4;
  attr.cap.max_send_sge = 1;
  attr.cap.max_recv_sge = 1;
  if (rdma_create_qp(c->id, c->pd, &attr)) FAIL_ERRNO("rdma_create_qp");
out:
  return rc;
}

// Register the caller's buffer and the little info buffer in one go.
static int register_memory(struct ctx *c, void *buf, size_t len, int access) {
  int rc = 0;
  c->data_mr = ibv_reg_mr(c->pd, buf, len, access);
  if (!c->data_mr)
    FAIL("ibv_reg_mr of %zu bytes: %s (memlock limit?)", len, strerror(errno));
  c->info_mr =
      ibv_reg_mr(c->pd, &c->info, sizeof(c->info), IBV_ACCESS_LOCAL_WRITE);
  if (!c->info_mr) FAIL_ERRNO("ibv_reg_mr info");
out:
  return rc;
}

// Wait for one completion of the given opcode: the standard
// poll / get_cq_event / ack / re-arm loop, blocking rather than spinning.
static int wait_completion(struct ctx *c, enum ibv_wc_opcode opcode) {
  int rc = 0;
  struct ibv_wc wc;
  while (1) {
    int n = ibv_poll_cq(c->cq, 1, &wc);
    if (n < 0) FAIL("ibv_poll_cq failed");
    if (n == 0) {
      struct ibv_cq *ev_cq;
      void *ev_ctx;
      if (ibv_get_cq_event(c->comp_ch, &ev_cq, &ev_ctx))
        FAIL_ERRNO("ibv_get_cq_event");
      ibv_ack_cq_events(ev_cq, 1);
      if (ibv_req_notify_cq(c->cq, 0)) FAIL_ERRNO("ibv_req_notify_cq");
      continue;
    }
    if (wc.status != IBV_WC_SUCCESS)
      FAIL("work completion %s (opcode %d)", ibv_wc_status_str(wc.status),
           wc.opcode);
    if (wc.opcode == opcode) break;
  }
out:
  return rc;
}

// the two-sided exchange that carries the region info
static int post_info_recv(struct ctx *c) {
  int rc = 0;
  struct ibv_sge sge = {(uint64_t)(uintptr_t)&c->info, sizeof(c->info),
                        c->info_mr->lkey};
  struct ibv_recv_wr wr = {0}, *bad;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  if (ibv_post_recv(c->id->qp, &wr, &bad)) FAIL_ERRNO("ibv_post_recv");
out:
  return rc;
}

static int post_info_send(struct ctx *c) {
  int rc = 0;
  struct ibv_sge sge = {(uint64_t)(uintptr_t)&c->info, sizeof(c->info),
                        c->info_mr->lkey};
  struct ibv_send_wr wr = {0}, *bad;
  wr.opcode = IBV_WR_SEND;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;
  if (ibv_post_send(c->id->qp, &wr, &bad)) FAIL_ERRNO("ibv_post_send");
out:
  return rc;
}

// one chunk of the transfer: read [raddr+off, +sz) into local+off
static int post_read(struct ctx *c, void *local, uint64_t raddr, uint32_t rkey,
                     size_t off, uint32_t sz) {
  int rc = 0;
  struct ibv_sge sge = {(uint64_t)(uintptr_t)local + off, sz,
                        c->data_mr->lkey};
  struct ibv_send_wr wr = {0}, *bad;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = raddr + off;
  wr.wr.rdma.rkey = rkey;
  if (ibv_post_send(c->id->qp, &wr, &bad)) FAIL_ERRNO("ibv_post_send");
out:
  return rc;
}

// one outstanding operation is enough: see the comment in rdma_fetch
static struct rdma_conn_param conn_param(void) {
  struct rdma_conn_param cp = {0};
  cp.responder_resources = 1;
  cp.initiator_depth = 1;
  cp.retry_count = 7;
  cp.rnr_retry_count = 7;
  return cp;
}

// reverse of build_connection, as the manual prescribes
static void destroy_connection(struct ctx *c) {
  if (c->id && c->id->qp) rdma_destroy_qp(c->id);
  if (c->data_mr) ibv_dereg_mr(c->data_mr);
  if (c->info_mr) ibv_dereg_mr(c->info_mr);
  if (c->cq) ibv_destroy_cq(c->cq);
  if (c->comp_ch) ibv_destroy_comp_channel(c->comp_ch);
  if (c->pd) ibv_dealloc_pd(c->pd);
  if (c->id) rdma_destroy_id(c->id);
  if (c->listen_id) rdma_destroy_id(c->listen_id);
  if (c->ec) rdma_destroy_event_channel(c->ec);
}

/* ------------------------------------------------------------------ */

// Expose buf for remote reads and block until the client disconnects.
// Prints one "listening" line so a driver script can tell when it is ready.
int rdma_serve(void *buf, size_t len, int port) {
  struct ctx c = {0};
  int rc = 0;

  c.ec = rdma_create_event_channel();
  if (!c.ec) FAIL_ERRNO("rdma_create_event_channel");
  if (rdma_create_id(c.ec, &c.listen_id, NULL, RDMA_PS_TCP))
    FAIL_ERRNO("rdma_create_id");

  struct sockaddr_in sin = {0};
  sin.sin_family = AF_INET;
  sin.sin_port = htons(port);
  sin.sin_addr.s_addr = INADDR_ANY;
  if (rdma_bind_addr(c.listen_id, (struct sockaddr *)&sin))
    FAIL_ERRNO("rdma_bind_addr");
  if (rdma_listen(c.listen_id, 1)) FAIL_ERRNO("rdma_listen");
  printf("[rdma] listening on port %d\n", port);
  fflush(stdout);

  rc = expect_cm_event(&c, RDMA_CM_EVENT_CONNECT_REQUEST, &c.id);
  if (rc) goto out;
  rc = build_connection(&c);
  if (rc) goto out;
  rc = register_memory(&c, buf, len,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
  if (rc) goto out;

  struct rdma_conn_param cp = conn_param();
  if (rdma_accept(c.id, &cp)) FAIL_ERRNO("rdma_accept");
  rc = expect_cm_event(&c, RDMA_CM_EVENT_ESTABLISHED, NULL);
  if (rc) goto out;

  // tell the client where to read from, then go idle
  c.info.addr = htonll((uint64_t)(uintptr_t)buf);
  c.info.len = htonll(len);
  c.info.rkey = htonl(c.data_mr->rkey);
  rc = post_info_send(&c);
  if (rc) goto out;
  rc = wait_completion(&c, IBV_WC_SEND);
  if (rc) goto out;

  // the HCA serves the reads from here on; this CPU does nothing
  rc = expect_cm_event(&c, RDMA_CM_EVENT_DISCONNECTED, NULL);

out:
  destroy_connection(&c);
  return rc;
}

// Pull len bytes from the server into buf. Stores the transfer time in *secs.
int rdma_fetch(const char *ip, int port, void *buf, size_t len, double *secs) {
  struct ctx c = {0};
  int rc = 0;

  c.ec = rdma_create_event_channel();
  if (!c.ec) FAIL_ERRNO("rdma_create_event_channel");
  if (rdma_create_id(c.ec, &c.id, NULL, RDMA_PS_TCP))
    FAIL_ERRNO("rdma_create_id");

  struct sockaddr_in sin = {0};
  sin.sin_family = AF_INET;
  sin.sin_port = htons(port);
  if (inet_pton(AF_INET, ip, &sin.sin_addr) != 1) FAIL("bad address '%s'", ip);
  if (rdma_resolve_addr(c.id, NULL, (struct sockaddr *)&sin, 3000))
    FAIL_ERRNO("rdma_resolve_addr");
  rc = expect_cm_event(&c, RDMA_CM_EVENT_ADDR_RESOLVED, NULL);
  if (rc) goto out;
  if (rdma_resolve_route(c.id, 3000)) FAIL_ERRNO("rdma_resolve_route");
  rc = expect_cm_event(&c, RDMA_CM_EVENT_ROUTE_RESOLVED, NULL);
  if (rc) goto out;

  rc = build_connection(&c);
  if (rc) goto out;
  rc = register_memory(&c, buf, len, IBV_ACCESS_LOCAL_WRITE);
  if (rc) goto out;
  // the receive must be posted before the connection is established
  rc = post_info_recv(&c);
  if (rc) goto out;

  struct rdma_conn_param cp = conn_param();
  if (rdma_connect(c.id, &cp)) FAIL_ERRNO("rdma_connect");
  rc = expect_cm_event(&c, RDMA_CM_EVENT_ESTABLISHED, NULL);
  if (rc) goto out;
  rc = wait_completion(&c, IBV_WC_RECV);
  if (rc) goto out;

  uint64_t raddr = ntohll(c.info.addr);
  uint64_t rlen = ntohll(c.info.len);
  uint32_t rkey = ntohl(c.info.rkey);
  if (rlen != len)
    FAIL("size mismatch: server has %lu bytes, client wants %zu", rlen, len);

  // One READ at a time, as in rping: 8 MiB takes ~800 us against a few us of
  // turnaround, so a deeper pipeline would buy about 1%.
  double t = now();
  for (size_t off = 0; off < len;) {
    uint32_t sz = (len - off < CHUNK) ? (uint32_t)(len - off) : CHUNK;
    rc = post_read(&c, buf, raddr, rkey, off, sz);
    if (rc) goto out;
    rc = wait_completion(&c, IBV_WC_RDMA_READ);
    if (rc) goto out;
    off += sz;
  }
  if (secs) *secs = now() - t;

  if (rdma_disconnect(c.id)) FAIL_ERRNO("rdma_disconnect");
  rc = expect_cm_event(&c, RDMA_CM_EVENT_DISCONNECTED, NULL);

out:
  destroy_connection(&c);
  return rc;
}
