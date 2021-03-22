# Pull base pytorch image
FROM pytorch/pytorch

# Install repository specific dependencies
RUN pip install scipy
RUN pip install pandas
RUN pip install sklearn
RUN pip install matplotlib
RUN pip install kneed

#  Define working directory
WORKDIR /workspace

# Define mountable volumes
VOLUME /workspace