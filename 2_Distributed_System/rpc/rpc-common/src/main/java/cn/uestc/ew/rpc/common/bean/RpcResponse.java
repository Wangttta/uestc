package cn.uestc.ew.rpc.common.bean;

import lombok.Getter;
import lombok.Setter;

/**
 * RPC 响应体，封装了一次 RPC 响应的结果，请求者及异常信息。
 */
@Getter
@Setter
public class RpcResponse {

    /**
     * 请求发起者
     */
    private String requestId;

    /**
     * 针对本次请求，返回处理过程中出现的异常信息
     */
    private Exception exception;

    /**
     * 针对本次请求，返回服务提供方的原始响应内容
     */
    private Object result;
}