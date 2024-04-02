package cn.uestc.ew.sample.rpc.server.service;

import cn.uestc.ew.rpc.server.RpcService;
import cn.uestc.ew.sample.rpc.api.StringService;
import org.apache.commons.lang3.StringUtils;

@RpcService(value = StringService.class)
public class StringServiceV1 implements StringService {

    @Override
    public String toUppercase(String text) {
        return StringUtils.upperCase(text);
    }
}