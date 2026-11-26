#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    uint32_t qpn;
    uint32_t psn;
    uint32_t rkey;
    uint64_t addr;
    uint8_t gid[16];
} RadRdmaInfo;

typedef struct {
    struct ibv_context* ctx;
    struct ibv_pd* pd;
    struct ibv_cq* scq;
    struct ibv_cq* rcq;
    struct ibv_qp* qp;
    struct ibv_mr* send_mr;
    struct ibv_mr* recv_mr;
    uint8_t* send_buf;
    uint8_t* recv_buf;
    void* send_dptr;
    void* recv_dptr;
    size_t buf_bytes;
    uint32_t remote_rkey;
    uint64_t remote_addr;
    uint8_t gid_index;
    uint64_t seq;
} RadRdmaCtx;

static int find_roce_v2_gid(struct ibv_context* ctx, uint8_t port, union ibv_gid* out_gid) {
    for (int i = 0; i < 16; ++i) {
        union ibv_gid g;
        if (ibv_query_gid(ctx, port, i, &g) != 0) continue;
        if (g.raw[10] == 0xff && g.raw[11] == 0xff && g.raw[12] == 10 && g.raw[13] == 100) {
            if (i == 3 || i == 2) {
                *out_gid = g;
                return i;
            }
        }
    }
    if (ibv_query_gid(ctx, port, 3, out_gid) == 0) return 3;
    if (ibv_query_gid(ctx, port, 1, out_gid) == 0) return 1;
    ibv_query_gid(ctx, port, 0, out_gid);
    return 0;
}

extern "C" {

void* rad_rdma_init(const char* dev_name, size_t buf_bytes, RadRdmaInfo* local_info) {
    int num_devs = 0;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devs);
    if (!dev_list || num_devs == 0) return NULL;

    struct ibv_device* chosen = NULL;
    for (int i = 0; i < num_devs; ++i) {
        const char* name = ibv_get_device_name(dev_list[i]);
        if (dev_name && strcmp(name, dev_name) == 0) {
            chosen = dev_list[i];
            break;
        }
    }
    if (!chosen) {
        for (int i = 0; i < num_devs; ++i) {
            struct ibv_context* tc = ibv_open_device(dev_list[i]);
            if (!tc) continue;
            struct ibv_port_attr pa;
            if (ibv_query_port(tc, 1, &pa) == 0 && pa.state == IBV_PORT_ACTIVE) {
                chosen = dev_list[i];
                ibv_close_device(tc);
                break;
            }
            ibv_close_device(tc);
        }
    }
    if (!chosen) {
        ibv_free_device_list(dev_list);
        return NULL;
    }

    RadRdmaCtx* r = (RadRdmaCtx*)calloc(1, sizeof(RadRdmaCtx));
    r->ctx = ibv_open_device(chosen);
    ibv_free_device_list(dev_list);
    if (!r->ctx) { free(r); return NULL; }

    r->pd = ibv_alloc_pd(r->ctx);
    r->scq = ibv_create_cq(r->ctx, 64, NULL, NULL, 0);
    r->rcq = ibv_create_cq(r->ctx, 64, NULL, NULL, 0);
    r->buf_bytes = buf_bytes;
    if (cudaHostAlloc((void**)&r->send_buf, buf_bytes, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc((void**)&r->recv_buf, buf_bytes, cudaHostAllocMapped) != cudaSuccess) {
        free(r);
        return NULL;
    }
    cudaHostGetDevicePointer(&r->send_dptr, r->send_buf, 0);
    cudaHostGetDevicePointer(&r->recv_dptr, r->recv_buf, 0);
    memset(r->send_buf, 0, buf_bytes);
    memset(r->recv_buf, 0, buf_bytes);

    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    r->send_mr = ibv_reg_mr(r->pd, r->send_buf, buf_bytes, access);
    r->recv_mr = ibv_reg_mr(r->pd, r->recv_buf, buf_bytes, access);

    struct ibv_qp_init_attr qia;
    memset(&qia, 0, sizeof(qia));
    qia.send_cq = r->scq;
    qia.recv_cq = r->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr = 32;
    qia.cap.max_recv_wr = 32;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 512;
    r->qp = ibv_create_qp(r->pd, &qia);
    if (!r->qp || !r->send_mr || !r->recv_mr) {
        free(r);
        return NULL;
    }

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = access;
    if (ibv_modify_qp(r->qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
        free(r);
        return NULL;
    }

    union ibv_gid gid;
    r->gid_index = (uint8_t)find_roce_v2_gid(r->ctx, 1, &gid);
    local_info->qpn = r->qp->qp_num;
    local_info->psn = 0x12345;
    local_info->rkey = r->recv_mr->rkey;
    local_info->addr = (uint64_t)(uintptr_t)r->recv_buf;
    memcpy(local_info->gid, gid.raw, 16);
    return (void*)r;
}

int rad_rdma_connect(void* handle, const RadRdmaInfo* remote_info) {
    RadRdmaCtx* r = (RadRdmaCtx*)handle;
    if (!r) return -1;
    r->remote_rkey = remote_info->rkey;
    r->remote_addr = remote_info->addr;

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote_info->qpn;
    attr.rq_psn = remote_info->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.grh.sgid_index = r->gid_index;
    attr.ah_attr.grh.hop_limit = 64;
    memcpy(attr.ah_attr.grh.dgid.raw, remote_info->gid, 16);

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(r->qp, &attr, flags) != 0) return -2;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = 0x12345;
    attr.max_rd_atomic = 1;
    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(r->qp, &attr, flags) != 0) return -3;

    for (int i = 0; i < 16; ++i) {
        struct ibv_recv_wr rwr;
        struct ibv_recv_wr* bad_rwr = NULL;
        memset(&rwr, 0, sizeof(rwr));
        rwr.wr_id = (uint64_t)i;
        if (ibv_post_recv(r->qp, &rwr, &bad_rwr) != 0) return -4;
    }
    return 0;
}

void* rad_rdma_send_dptr(void* handle) {
    RadRdmaCtx* r = (RadRdmaCtx*)handle;
    return r ? r->send_dptr : NULL;
}

void* rad_rdma_recv_dptr(void* handle) {
    RadRdmaCtx* r = (RadRdmaCtx*)handle;
    return r ? r->recv_dptr : NULL;
}

int rad_rdma_exchange_mapped(void* handle, int n_bytes) {
    RadRdmaCtx* r = (RadRdmaCtx*)handle;
    if (!r || (size_t)n_bytes > r->buf_bytes) return -1;

    struct ibv_sge sge;
    sge.addr = (uint64_t)(uintptr_t)r->send_buf;
    sge.length = (uint32_t)n_bytes;
    sge.lkey = r->send_mr->lkey;

    struct ibv_send_wr wr;
    struct ibv_send_wr* bad_wr = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = ++r->seq;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED | ((n_bytes <= 512) ? IBV_SEND_INLINE : 0);
    wr.imm_data = (uint32_t)r->seq;
    wr.wr.rdma.remote_addr = r->remote_addr;
    wr.wr.rdma.rkey = r->remote_rkey;
    if (ibv_post_send(r->qp, &wr, &bad_wr) != 0) return -2;

    struct ibv_wc wc;
    while (ibv_poll_cq(r->rcq, 1, &wc) == 0) {}
    if (wc.status != IBV_WC_SUCCESS) return -4;
    struct ibv_recv_wr rwr;
    struct ibv_recv_wr* bad_rwr = NULL;
    memset(&rwr, 0, sizeof(rwr));
    rwr.wr_id = wc.wr_id;
    ibv_post_recv(r->qp, &rwr, &bad_rwr);

    while (ibv_poll_cq(r->scq, 1, &wc) == 0) {}
    if (wc.status != IBV_WC_SUCCESS) return -3;
    return 0;
}

void* rad_rdma_send_hptr(void* handle) {
    RadRdmaCtx* r = (RadRdmaCtx*)handle;
    return r ? (void*)r->send_buf : NULL;
}

void* rad_rdma_recv_hptr(void* handle) {
    RadRdmaCtx* r = (RadRdmaCtx*)handle;
    return r ? (void*)r->recv_buf : NULL;
}

void rad_rdma_close(void* handle) {
    RadRdmaCtx* r = (RadRdmaCtx*)handle;
    if (!r) return;
    if (r->qp) ibv_destroy_qp(r->qp);
    if (r->scq) ibv_destroy_cq(r->scq);
    if (r->rcq) ibv_destroy_cq(r->rcq);
    if (r->send_mr) ibv_dereg_mr(r->send_mr);
    if (r->recv_mr) ibv_dereg_mr(r->recv_mr);
    if (r->pd) ibv_dealloc_pd(r->pd);
    if (r->ctx) ibv_close_device(r->ctx);
    if (r->send_buf) cudaFreeHost(r->send_buf);
    if (r->recv_buf) cudaFreeHost(r->recv_buf);
    free(r);
}

}
