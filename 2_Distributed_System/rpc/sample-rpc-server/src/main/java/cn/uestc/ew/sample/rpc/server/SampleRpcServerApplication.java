package cn.uestc.ew.sample.rpc.server;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ImportResource;

@SpringBootApplication
@ImportResource(locations = {"classpath:spring.xml"})
public class SampleRpcServerApplication {

	public static void main(String[] args) {
		SpringApplication.run(SampleRpcServerApplication.class, args);
	}
}