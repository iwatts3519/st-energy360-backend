# Set up base environment
FROM python:3.9

# Expose Ports
ENV LISTEN_PORT=80
EXPOSE 80

#update environment
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install -U pip

# Set working directory
WORKDIR /RAFA
# Install Tim
COPY ./setup.py ./setup.py
COPY ./tim_client ./tim_client
COPY requirements.txt ./requirements.txt
CMD ["python", "setup.py install"]

#Install Requirements
COPY requirements2.txt ./requirements2.txt
RUN pip install -r requirements2.txt

# Copy Files to container image
COPY . .
# start app
CMD ["python", "keele_data.py"]

