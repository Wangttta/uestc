package cn.uestc.ew.sample.rpc.client;

import cn.uestc.ew.rpc.client.RpcClientProxy;
import cn.uestc.ew.sample.rpc.api.MathService;
import cn.uestc.ew.sample.rpc.api.StringService;
import cn.uestc.ew.sample.rpc.api.TimeoutService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
@RequestMapping("/test")
@ResponseBody
public class TestController {

    @Autowired
    private RpcClientProxy proxy;

    @GetMapping("/sum/int")
    public String testIntSum(int a, int b) {
        MathService mathService = proxy.create(MathService.class);
        int sum = mathService.sum(a, b);
        return String.format("测试整数加法：%d + %d = %d", a, b, sum);
    }

    @GetMapping("/sum/float")
    public String testFloatSum(float a, float b) {
        MathService mathService = proxy.create(MathService.class);
        float sum = mathService.sum(a, b);
        return String.format("测试浮点数加法：%f + %f = %f", a, b, sum);
    }

    @GetMapping("/string")
    public String testString(String text) {
        StringService stringService = proxy.create(StringService.class);
        String upperText = stringService.toUppercase(text);
        return String.format("测试字符串转换大写：%s => %s", text, upperText);
    }

    @GetMapping("/timeout")
    public String testTimeout() {
        TimeoutService timeoutService = proxy.create(TimeoutService.class);
        String resp = timeoutService.timeoutAtFirstTime();
        return String.format("测试超时重试接口：'' => '%s'", resp);
    }
}