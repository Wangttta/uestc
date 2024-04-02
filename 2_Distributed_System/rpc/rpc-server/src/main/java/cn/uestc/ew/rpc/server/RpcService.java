package cn.uestc.ew.rpc.server;

import org.springframework.stereotype.Component;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * RPC 服务描述注解，用于标注所有可提供 RPC 服务的本地对象
 */
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Component
public @interface RpcService {

    /**
     * 服务接口类型
     */
    Class<?> value();

    /**
     * 服务版本号
     */
    String version() default "";
}