This code is for paper *Movie Fill in the Blank with Adaptive Temporal Attention and Description Update*. This paper aims at tackling [Movie Fill-in-the-Blank](https://sites.google.com/site/describingmovies/lsmdc-2016/movie-fill-in-the-blank) task.

To run the code, you need:
> * Lasagne 0.2
> * Theano 0.9
> * Caffe

1. Download [downloadChallengeData.sh](http://datasets.d2.mpi-inf.mpg.de/movieDescription/protected/lsmdc2016/downloadChallengeData.sh). Then run this script.
    <pre class=”brush: shell; gutter: true;”>
    > bash downloadChallengeData.sh
    </pre>

2. Enter ***extractFeature*** folder, change the variable ***pathPrefix*** of ***getVideoFilePath.py*** to your path to the folder where video files are.
    <pre class=”brush: shell; gutter: true;”>
    > python getVideoFilePath.py
    </pre>

3. Run ***extract_feature.py***
    * Download [ResNet-152-deploy.prototxt](https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt)
    * Download [ResNet_mean.binaryproto](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
    * Download [ResNet-152-model.caffemodel](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
    
    Copy them to ***extractFeature/models***
    <pre class=”brush: shell; gutter: true;”>
    > python extract_feature.py
    </pre>

4. Change the config file ***config.py*** and run ***fib.py***.
    * ***video_data_dir***: path to video features folder, i.e., the  absolute path of ***featrure/pool5***.
    * ***text_data_dir***: path to data file folder, i.e., the absolute path of ***data***.
    * ***word2vec_model_dir***: path to word2vec model
    * ***modelsave_dir***: path to save models.
    * ***performance_dir***: path to file recording performance.

    If you want to train the model:
    <pre class=”brush: shell; gutter: true;”>
    > python fib.py
    </pre>

    Test model
    * Modify ***fib.py***
    <pre class=”brush: python; gutter: true;”>
    train = False
    </pre>
    and modify ***modelidx*** to the saved model index.
