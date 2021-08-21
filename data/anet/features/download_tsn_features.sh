# Download TSN feature files
wget http://youcook2.eecs.umich.edu/static/dat/anet_densecap/training_feat_anet.tar.gz
wget http://youcook2.eecs.umich.edu/static/dat/anet_densecap/validation_feat_anet.tar.gz
wget http://youcook2.eecs.umich.edu/static/dat/anet_densecap/testing_feat_anet.tar.gz

tar xvzf training_feat_anet.tar.gz
tar xvzf validation_feat_anet.tar.gz
tar xvzf testing_feat_anet.tar.gz
mkdir resnet_bn
mv testing/* resnet_bn
mv training/* resnet_bn
mv validation/* resnet_bn
