package cn.uestc.ew.sample.rpc.client;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ImportResource;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
@SpringBootApplication
@ImportResource(locations = {"classpath:spring.xml"})
public class SampleRpcClientApplication {

	@GetMapping
	@ResponseBody
	public String index() {
		return "OK";
	}

	public static void main(String[] args) {
		SpringApplication.run(SampleRpcClientApplication.class, args);
	}
}