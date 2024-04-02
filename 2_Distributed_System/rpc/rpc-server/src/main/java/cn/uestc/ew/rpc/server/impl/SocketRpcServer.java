package cn.uestc.ew.rpc.server.impl;

import cn.uestc.ew.rpc.common.bean.RpcRequest;
import cn.uestc.ew.rpc.common.bean.RpcResponse;
import cn.uestc.ew.rpc.common.util.SerializationUtils;
import cn.uestc.ew.rpc.common.util.codec.RpcDecoder;
import cn.uestc.ew.rpc.registry.ServiceRegistry;
import cn.uestc.ew.rpc.server.RpcServer;
import cn.uestc.ew.rpc.server.RpcServerHandler;
import cn.uestc.ew.rpc.server.RpcService;
import com.alibaba.fastjson2.JSON;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.HashMap;
import java.util.Map;

/**
 * RPC 服务端，启动服务端监听，基于 Socket BIO 实现
 */
@Slf4j
public class SocketRpcServer implements ApplicationContextAware, InitializingBean, RpcServer {

    /**
     * 服务端地址
     */
    private final String serverAddress;

    /**
     * 服务注册工具
     */
    private final ServiceRegistry serviceRegistry;

    /**
     * 存放 "服务名" 与 "服务" 之间的映射
     */
    private final Map<String, Object> handlerMap = new HashMap<>();

    /**
     * RPC 请求处理器
     */
    private final RpcServerHandler serverHandler = new RpcServerHandler(handlerMap);

    public SocketRpcServer(String serverAddress, ServiceRegistry serviceRegistry) {
        this.serverAddress = serverAddress;
        this.serviceRegistry = serviceRegistry;
    }

    /**
     * 应用启动时，提供 Spring 提供的注解扫描，自动收集所有 RPC 服提供者，
     * 将其注册到 {@link RpcServerHandler} 的 {@code handleMap} 中
     *
     * @param ctx Spring 上下文
     */
    @Override
    public void setApplicationContext(ApplicationContext ctx) {

        // 1. 扫描带有 RpcService 注解的类
        Map<String, Object> serviceBeanMap = ctx.getBeansWithAnnotation(RpcService.class);
        if (MapUtils.isEmpty(serviceBeanMap)) return;

        // 2. 初始化 HandlerMap 对象
        for (Object serviceBean : serviceBeanMap.values()) {
            RpcService rpcService = serviceBean.getClass().getAnnotation(RpcService.class);
            String serviceName = rpcService.value().getName();
            String serviceVersion = rpcService.version();
            if (StringUtils.isNotEmpty(serviceVersion)) {
                serviceName += "-" + serviceVersion;
            }
            handlerMap.put(serviceName, serviceBean);
        }
    }

    /**
     * RPC 服务端启动时，在本地
     */
    @Override
    public void afterPropertiesSet() {

        // 1. 注册 RPC 服务到注册中心
        if (serviceRegistry != null) {
            for (String interfaceName : handlerMap.keySet()) {
                serviceRegistry.register(interfaceName, serverAddress);
                log.debug("Register service: name={}, address={}", interfaceName, serverAddress);
            }
        }

        // 2. 启动 RPC Server Stub
        this.startup();
    }

    @Override
    public void startup() {

        // 1. 获取 RPC 服务的端口号并启动监听
        String[] addressArray = StringUtils.split(serverAddress, ":");
        int port = Integer.parseInt(addressArray[1]);

        // 2. 启动服务器监听
        try (ServerSocket socket = new ServerSocket(port)) {
            while (true) {
                // 2.1 服务器阻塞等待客户端请求
                try (Socket client = socket.accept();
                     InputStream in = client.getInputStream();
                     OutputStream out = client.getOutputStream()) {

                    // 2.2 读取请求体
                    int requestSize = RpcDecoder.toInt(in.readNBytes(4));   // 请求体大小
                    byte[] requestBytes = in.readNBytes(requestSize);
                    RpcRequest rpcRequest = SerializationUtils.deserialize(requestBytes, RpcRequest.class);

                    // 2.3 调用本地服务处理请求
                    log.info("RPC: request body = {}", JSON.toJSONString(rpcRequest));
                    RpcResponse rpcResponse = serverHandler.handle(rpcRequest);
                    log.info("RPC: response body = {}", JSON.toJSONString(rpcResponse));

                    // 2.4 写响应并关闭客户端连接
                    out.write(SerializationUtils.serialize(rpcResponse));
                    out.flush();
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}