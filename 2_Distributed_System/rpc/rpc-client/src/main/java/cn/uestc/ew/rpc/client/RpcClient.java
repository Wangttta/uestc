package cn.uestc.ew.rpc.client;

import cn.uestc.ew.rpc.common.bean.RpcRequest;
import cn.uestc.ew.rpc.common.bean.RpcResponse;
import cn.uestc.ew.rpc.common.config.RpcConfig;

public interface RpcClient {

    /**
     * RPC 客户端发送方法，可基于不同的网络通信技术实现如 Socket BIO、Socket NIO、Netty 等
     */
    RpcResponse send(RpcRequest request) throws Exception;

    /**
     * 注入 RPC 配置文件
     * @param config 应用提供的 RPC 配置文件
     */
    void setRpcConfig(RpcConfig config);
}