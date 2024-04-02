package cn.uestc.ew.rpc.common.bean;

import lombok.Getter;
import lombok.Setter;

/**
 * RPC 请求体，封装了一次 RPC 请求的来源、目的及参数信息。
 */
@Getter
@Setter
public class RpcRequest {

    /**
     * 请求者标识符
     */
    private String requestId;

    /**
     * 重试次数
     */
    private int retryTimes;

    /**
     * 请求接口名称
     */
    private String interfaceName;

    /**
     * 请求服务版本
     */
    private String serviceVersion;

    /**
     * 请求方法名称
     */
    private String methodName;

    /**
     * 参数列表
     */
    private Object[] parameters;

    /**
     * 参数类型列表，与参数列表一一对应
     */
    private Class<?>[] parameterTypes;
}