package cn.uestc.ew.sample.rpc.server.service;

import cn.uestc.ew.rpc.server.RpcService;
import cn.uestc.ew.sample.rpc.api.MathService;

@RpcService(value = MathService.class)
public class MathServiceV1 implements MathService {

    @Override
    public int sum(int a, int b) {
        return a + b;
    }

    @Override
    public float sum(float a, float b) {
        return a + b;
    }
}