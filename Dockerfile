FROM python:latest
MAINTAINER Robin Le Bihan

COPY requirements.txt /tmp/

RUN python3 -m venv /env
RUN /env/bin/pip install -r /tmp/requirements.txt

COPY src/toSubmit.py /

VOLUME /datas
VOLUME /output

CMD ["/env/bin/python", "/toSubmit.py", "/datas"]
