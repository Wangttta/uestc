FROM openjdk:19
MAINTAINER Yang XiangDong <yangxiangdong.cs@aliyun.com>

EXPOSE 8080
# Get parameter JAR_FILE
ARG JAR_FILE
# Copy and Run
ADD /${JAR_FILE} /${JAR_FILE}
ENTRYPOINT Java -jar ${JAR_FILE}