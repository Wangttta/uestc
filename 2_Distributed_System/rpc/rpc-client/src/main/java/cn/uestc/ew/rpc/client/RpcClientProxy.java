package cn.uestc.ew.rpc.client;

import cn.uestc.ew.rpc.client.impl.SocketRpcClient;
import cn.uestc.ew.rpc.common.bean.RpcRequest;
import cn.uestc.ew.rpc.common.bean.RpcResponse;
import cn.uestc.ew.rpc.common.config.RpcConfig;
import cn.uestc.ew.rpc.common.exception.Asserts;
import cn.uestc.ew.rpc.registry.ServiceDiscovery;
import com.alibaba.fastjson2.JSON;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.net.SocketTimeoutException;
import java.util.Objects;
import java.util.UUID;

/**
 * RPC 代理，也就是 RPC Stub。为客户端提供服务调用、编解码及远程调用结果返回，
 * 屏蔽了远程过程调用涉及的网络通信。
 */
@Slf4j
public class RpcClientProxy {

    /**
     * RPC 参数
     */
    private final RpcConfig rpcConfig;

    /**
     * 服务发现工具
     */
    private final ServiceDiscovery serviceDiscovery;

    public RpcClientProxy(ServiceDiscovery serviceDiscovery, RpcConfig rpcConfig) {
        this.serviceDiscovery = serviceDiscovery;
        this.rpcConfig = rpcConfig;
    }

    /**
     * 根据需要访问的 API 接口类，创建代理服务对象，使用默认版本号（""）。
     *
     * @see RpcClientProxy#create(Class, String)
     */
    public <T> T create(final Class<?> interfaceClass) {
        return create(interfaceClass, StringUtils.EMPTY);
    }

    /**
     * 根据需要访问的 API 接口类及服务版本号，创建代理服务对象。
     *
     * <p>此接口一般由服务方提供，并不是真正的服务对象，而是对外发布的一个
     * 函数标识符与真正的服务对象相一致的空的代理接口。</p>
     *
     * @param interfaceClass 需要访问的 API 接口类，用于获取服务相关的信息如地址、端口等
     * @param serviceVersion 需要访问的服务版本号
     * @return 目标服务代理接口
     * @param <T> 需要访问的 API 接口类
     */
    @SuppressWarnings("unchecked")
    public <T> T create(final Class<?> interfaceClass, final String serviceVersion) {
        // 创建动态代理对象
        return (T) Proxy.newProxyInstance(
                interfaceClass.getClassLoader(),
                new Class<?>[]{interfaceClass},
                new RpcInvocationHandler(interfaceClass, serviceVersion));
    }

    /**
     * RPC 服务代理处理器，用于规定代理对象如何处理服务调用及返回，每次发起
     * 服务调用，都会动态创建一个服务代理对象。而代理对象由于访问的服务不同，
     * 需要不同的代理处理器，来完成自己需要执行的操作。
     */
    private class RpcInvocationHandler implements InvocationHandler {

        /**
         * 本次请求需要访问的 API 接口类
         */
        private final Class<?> interfaceClass;

        /**
         * 本次请求需要访问的 API 版本号
         */
        private final String serviceVersion;

        RpcInvocationHandler (final Class<?> interfaceClass, final String serviceVersion) {
            this.interfaceClass = interfaceClass;
            this.serviceVersion = serviceVersion;
        }

        /**
         * 此方法规定了 RPC 服务代理类的执行逻辑
         *
         * <p>RPC 框架下，客户端代理接收客户发起的服务调用请求，为其发起远程调用，
         * 屏蔽远程通信面临的问题，其顺序处理流程如下：</p>
         * <ol>
         *     <li>根据客户端参数查找远程服务的地址</li>
         *     <li>创建请求体并发起网络请求</li>
         *     <li>接收并处理返回结果，将其返回给客户端</li>
         * </ol>
         *
         * @return RPC 服务代理类的执行结果，正常情况下应该是调用远程服务返回的内容，
         *         被正确解码后的结果
         * @throws Exception 代理类执行过程中可能出现的任何异常，如网络通信异常等
         */
        @Override
        public Object invoke(Object proxy, Method method, Object[] args) throws Exception {

            // 1. 创建 RPC 请求对象并设置请求属性
            RpcRequest request = new RpcRequest();
            request.setRequestId(UUID.randomUUID().toString());
            request.setInterfaceName(method.getDeclaringClass().getName());
            request.setMethodName(method.getName());
            request.setParameterTypes(method.getParameterTypes());
            request.setParameters(args);
            request.setServiceVersion(serviceVersion);

            // 2. 获取 RPC 服务地址
            String serviceName = interfaceClass.getName();
            if (StringUtils.isNotEmpty(serviceVersion)) {
                serviceName += "-" + serviceVersion;
            }
            String serviceAddress = serviceDiscovery.discover(serviceName);
            Asserts.notEmpty(serviceAddress, String.format("Service address of [%s] is empty", serviceName));
            log.debug("Discover service: name={}, address={}.", serviceName, serviceAddress);
            String[] array = StringUtils.split(serviceAddress, ":");
            String host = array[0];     // 服务端 IP
            int port = Integer.parseInt(array[1]); // 服务端端口

            // 4. 创建 RPC 客户端对象并发送 RPC 请求
            log.info("RPC: request body = {}", JSON.toJSONString(request));
            RpcClient client = new SocketRpcClient(host, port, rpcConfig);
            long time = System.currentTimeMillis();
            RpcResponse response = sendWithRetry(client, request);
            Asserts.notNull(response, String.format("Cannot receive any response from [%s]", serviceAddress));
            log.info("RPC: resp body = {}, waste time = {} ms", JSON.toJSONString(response), System.currentTimeMillis() - time);

            // 5. 返回 RPC 响应结果
            if (Objects.nonNull(response.getException())) {
                throw response.getException();
            }
            return response.getResult();
        }

        /**
         * 根据重试配置，决定以何种语义（"至多一次"，"至少一次"）发送请求
         * @see RpcClient#send(RpcRequest)
         */
        private RpcResponse sendWithRetry(RpcClient client, RpcRequest request) throws Exception {

            // No retry (At-most-once)
            if (rpcConfig == null || rpcConfig.noRetry()) {
                return client.send(request);
            }

            // Send with retry (At-least-once)
            try {
                request.setRetryTimes(1);
                return client.send(request);
            } catch (SocketTimeoutException e) {
                log.info("RPC: service timeout, start retry...");
                int i = 1;
                while (rpcConfig.needRetry(i++)) {
                    try {
                        request.setRetryTimes(i);
                        log.info("RPC: retry {}th, request body = {}", i, JSON.toJSONString(request));
                        return client.send(request);
                    } catch (SocketTimeoutException ee) {
                        // Ignore
                        log.info("RPC: retry {}th, exception = {}", i, ee.getMessage());
                    }
                }
            }
            log.info("RPC: retry over, service unavailable");
            return null;
        }
    }
}