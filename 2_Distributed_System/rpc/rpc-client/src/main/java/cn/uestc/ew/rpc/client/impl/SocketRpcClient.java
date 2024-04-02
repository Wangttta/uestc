package cn.uestc.ew.rpc.client.impl;

import cn.uestc.ew.rpc.client.RpcClient;
import cn.uestc.ew.rpc.common.bean.RpcRequest;
import cn.uestc.ew.rpc.common.bean.RpcResponse;
import cn.uestc.ew.rpc.common.config.RpcConfig;
import cn.uestc.ew.rpc.common.util.SerializationUtils;
import cn.uestc.ew.rpc.common.util.codec.RpcEncoder;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;

/**
 * RPC 客户端，用于 RPC 服务代理发起远程调用时，处理相关的网络调用逻辑，
 * 基于 Socket BIO 实现
 */
@Slf4j
public class SocketRpcClient implements RpcClient {

    /**
     * 远程服务地址
     */
    private final String host;

    /**
     * 远程服务端口
     */
    private final int port;

    /**
     * RPC 配置文件
     */
    private RpcConfig config;

    @Override
    public void setRpcConfig(RpcConfig config) {
        this.config = config;
    }

    public SocketRpcClient(String host, int port, RpcConfig config) {
        this.host = host;
        this.port = port;
        this.config = config;
    }

    /**
     * 向远程服务发起请求，基于 Socket 实现的阻塞通信
     *
     * @param request RPC 请求体
     * @return RPC 响应体
     */
    public RpcResponse send(RpcRequest request) throws IOException {
        byte[] requestBytes = SerializationUtils.serialize(request);
        try (Socket socket  = new Socket(host, port);
             OutputStream out = socket.getOutputStream();
             InputStream in = socket.getInputStream()) {

            // 1. 封装请求参数（4B + RpcRequest）
            socket.setSoTimeout(config.getTimeout());
            out.write(RpcEncoder.toBytes(requestBytes.length));
            out.write(requestBytes);
            socket.getOutputStream().flush();

            // 2. 接收服务端响应
            byte[] respBytes = in.readAllBytes();   // 阻塞
            socket.getInputStream().close();
            return SerializationUtils.deserialize(respBytes, RpcResponse.class);
        }
    }
}