# Download C3D feature files
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-00
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-01
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-02
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-03
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-04
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-05
cat activitynet_v1-3.part-* > c3d_features.zip && unzip c3d_features.zip
python convert_c3d_h5_to_npy.py