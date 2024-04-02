package cn.uestc.ew.sample.rpc.server.service;

import cn.uestc.ew.rpc.server.RpcService;
import cn.uestc.ew.sample.rpc.api.TimeoutService;

@RpcService(value = TimeoutService.class)
public class TimeoutServiceV1 implements TimeoutService {

    private static int counter = 0;

    @Override
    public String timeoutAtFirstTime() {
        if (counter++ < 5) {
            try {
                Thread.sleep(5000);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        return String.format("TimeoutServiceV1 收到的第 %d 次请求", counter);
    }
}