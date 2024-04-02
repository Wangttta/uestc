package cn.uestc.ew.rpc.server;

import cn.uestc.ew.rpc.common.bean.RpcRequest;
import cn.uestc.ew.rpc.common.bean.RpcResponse;
import cn.uestc.ew.rpc.server.impl.SocketRpcServer;
import lombok.extern.slf4j.Slf4j;
import net.sf.cglib.reflect.FastClass;
import net.sf.cglib.reflect.FastMethod;
import org.apache.commons.lang3.StringUtils;

import java.util.Map;

/**
 * RPC 服务端处理器，规定了服务端收到远程调用请求时的处理逻辑，
 * 即对于 {@link SocketRpcServer} 收到的调用请求，由当前类负责处
 * 理，也就是执行真正的服务调用并返回原始结果。
 */
@Slf4j
public class RpcServerHandler {

    /**
     * 当前服务器中，服务名称（服务类名称-服务版本号）与服务类（Service Bean）的映射
     */
    private final Map<String, Object> handlerMap;

    public RpcServerHandler(Map<String, Object> handlerMap) {
        this.handlerMap = handlerMap;
    }

    /**
     * 在客户端的输入到达后，经过一系列网络相关的前置处理，取得
     * {@link RpcRequest} 并调用当前方法，执行业务处理逻辑
     */
    public RpcResponse handle(RpcRequest request) {
        // 1. 创建并初始化 RPC 响应对象
        RpcResponse response = new RpcResponse();
        response.setRequestId(request.getRequestId());
        try {
            // 2. 服务端执行本地调用
            Object result = call(request);
            // 3. 返回调用结果
            response.setResult(result);
            return response;
        } catch (Exception e) {
            log.error("Server error occurred during local procedure call", e);
            response.setException(e);
            throw new RuntimeException(e);
        }
    }

    /**
     * 自定义服务端本地过程调用逻辑，处理流程如下：
     * <ol>
     *     <li>获取提供服务的 Bean 对象</li>
     *     <li>反射调用其指定方法（RpcRequest 指定）</li>
     * </ol>
     *
     * @param request 客户端调用信息，包括调用者身份、目标服务及参数等
     * @return 服务端执行本地调用后的原始返回结果
     * @throws Exception 反射调用时可能出现的任何异常
     */
    private Object call(RpcRequest request) throws Exception {
        // 1. 从容器中获取本地服务对象
        String serviceName = request.getInterfaceName();
        String serviceVersion = request.getServiceVersion();
        if (StringUtils.isNotEmpty(serviceVersion)) {
            serviceName += "-" + serviceVersion;
        }
        Object serviceBean = handlerMap.get(serviceName);
        if (serviceBean == null) {
            throw new RuntimeException(String.format("Can not find service bean by service name: %s", serviceName));
        }
        // 2. 获取反射调用所需的参数
        Class<?> serviceClass = serviceBean.getClass();
        String methodName = request.getMethodName();
        Class<?>[] parameterTypes = request.getParameterTypes();
        Object[] parameters = request.getParameters();
        // 3. 使用 CGLib 执行反射调用
        FastClass serviceFastClass = FastClass.create(serviceClass);
        FastMethod serviceFastMethod = serviceFastClass.getMethod(methodName, parameterTypes);
        return serviceFastMethod.invoke(serviceBean, parameters);
    }
}