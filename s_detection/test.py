import requests

url = 'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
response = requests.get(url, stream=True)

with open('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5', 'wb') as f:
    f.write(response.content)