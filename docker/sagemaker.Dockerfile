FROM lmsysorg/sglang:latest

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY serve /usr/bin/serve
RUN chmod 777 /usr/bin/serve

ENTRYPOINT [ "/usr/bin/serve" ]
