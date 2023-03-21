from kaffe.tensorflow import Network

class MyNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, biased=False, relu=False, name=conv1_7x7_s2)
             .batch_normalization(scale_offset=False, relu=True, name=conv1_7x7_s2_bn)
             .max_pool(3, 3, 2, 2, name=pool1_3x3_s2)
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name=conv2_3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=conv2_3x3_reduce_bn)
             .conv(3, 3, 192, 1, 1, biased=False, relu=False, name=conv2_3x3)
             .batch_normalization(scale_offset=False, relu=True, name=conv2_3x3_bn)
             .max_pool(3, 3, 2, 2, name=pool2_3x3_s2)
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name=inception_3a_1x1)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3a_1x1_bn))

        (self.feed('pool2_3x3_s2')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name=inception_3a_3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3a_3x3_reduce_bn)
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name=inception_3a_3x3)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3a_3x3_bn))

        (self.feed('pool2_3x3_s2')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name=inception_3a_double3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3a_double3x3_reduce_bn)
             .conv(3, 3, 96, 1, 1, biased=False, relu=False, name=inception_3a_double3x3a)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3a_double3x3a_bn)
             .conv(3, 3, 96, 1, 1, biased=False, relu=False, name=inception_3a_double3x3b)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3a_double3x3b_bn))

        (self.feed('pool2_3x3_s2')
             .avg_pool(3, 3, 1, 1, name=inception_3a_pool)
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, name=inception_3a_pool_proj)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3a_pool_proj_bn))

        (self.feed('inception_3a_1x1_bn', 
                   'inception_3a_3x3_bn', 
                   'inception_3a_double3x3b_bn', 
                   'inception_3a_pool_proj_bn')
             .concat(3, name=inception_3a_output)
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name=inception_3b_1x1)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3b_1x1_bn))

        (self.feed('inception_3a_output')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name=inception_3b_3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3b_3x3_reduce_bn)
             .conv(3, 3, 96, 1, 1, biased=False, relu=False, name=inception_3b_3x3)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3b_3x3_bn))

        (self.feed('inception_3a_output')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name=inception_3b_double3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3b_double3x3_reduce_bn)
             .conv(3, 3, 96, 1, 1, biased=False, relu=False, name=inception_3b_double3x3a)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3b_double3x3a_bn)
             .conv(3, 3, 96, 1, 1, biased=False, relu=False, name=inception_3b_double3x3b)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3b_double3x3b_bn))

        (self.feed('inception_3a_output')
             .avg_pool(3, 3, 1, 1, name=inception_3b_pool)
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name=inception_3b_pool_proj)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3b_pool_proj_bn))

        (self.feed('inception_3b_1x1_bn', 
                   'inception_3b_3x3_bn', 
                   'inception_3b_double3x3b_bn', 
                   'inception_3b_pool_proj_bn')
             .concat(3, name=inception_3b_output)
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name=inception_3c_3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3c_3x3_reduce_bn)
             .conv(3, 3, 160, 2, 2, biased=False, relu=False, name=inception_3c_3x3)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3c_3x3_bn))

        (self.feed('inception_3b_output')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name=inception_3c_double3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3c_double3x3_reduce_bn)
             .conv(3, 3, 96, 1, 1, biased=False, relu=False, name=inception_3c_double3x3a)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3c_double3x3a_bn)
             .conv(3, 3, 96, 2, 2, biased=False, relu=False, name=inception_3c_double3x3b)
             .batch_normalization(scale_offset=False, relu=True, name=inception_3c_double3x3b_bn))

        (self.feed('inception_3b_output')
             .max_pool(3, 3, 2, 2, name=inception_3c_pool))

        (self.feed('inception_3c_pool', 
                   'inception_3c_3x3_bn', 
                   'inception_3c_double3x3b_bn')
             .concat(3, name=inception_3c_output)
             .avg_pool(5, 5, 3, 3, padding='VALID', name=pool3_5x5_s3)
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name=loss1_conv)
             .batch_normalization(scale_offset=False, relu=True, name=loss1_conv_bn)
             .fc(1024, relu=False, name=loss1_fc)
             .batch_normalization(scale_offset=False, relu=True, name=loss1_fc_bn)
             .fc(1000, relu=False, name=loss1_classifier))

        (self.feed('inception_3c_output')
             .conv(1, 1, 224, 1, 1, biased=False, relu=False, name=inception_4a_1x1)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4a_1x1_bn))

        (self.feed('inception_3c_output')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name=inception_4a_3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4a_3x3_reduce_bn)
             .conv(3, 3, 96, 1, 1, biased=False, relu=False, name=inception_4a_3x3)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4a_3x3_bn))

        (self.feed('inception_3c_output')
             .conv(1, 1, 96, 1, 1, biased=False, relu=False, name=inception_4a_double3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4a_double3x3_reduce_bn)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name=inception_4a_double3x3a)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4a_double3x3a_bn)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name=inception_4a_double3x3b)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4a_double3x3b_bn))

        (self.feed('inception_3c_output')
             .avg_pool(3, 3, 1, 1, name=inception_4a_pool)
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name=inception_4a_pool_proj)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4a_pool_proj_bn))

        (self.feed('inception_4a_1x1_bn', 
                   'inception_4a_3x3_bn', 
                   'inception_4a_double3x3b_bn', 
                   'inception_4a_pool_proj_bn')
             .concat(3, name=inception_4a_output)
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name=inception_4b_1x1)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4b_1x1_bn))

        (self.feed('inception_4a_output')
             .conv(1, 1, 96, 1, 1, biased=False, relu=False, name=inception_4b_3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4b_3x3_reduce_bn)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name=inception_4b_3x3)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4b_3x3_bn))

        (self.feed('inception_4a_output')
             .conv(1, 1, 96, 1, 1, biased=False, relu=False, name=inception_4b_double3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4b_double3x3_reduce_bn)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name=inception_4b_double3x3a)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4b_double3x3a_bn)
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name=inception_4b_double3x3b)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4b_double3x3b_bn))

        (self.feed('inception_4a_output')
             .avg_pool(3, 3, 1, 1, name=inception_4b_pool)
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name=inception_4b_pool_proj)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4b_pool_proj_bn))

        (self.feed('inception_4b_1x1_bn', 
                   'inception_4b_3x3_bn', 
                   'inception_4b_double3x3b_bn', 
                   'inception_4b_pool_proj_bn')
             .concat(3, name=inception_4b_output)
             .conv(1, 1, 160, 1, 1, biased=False, relu=False, name=inception_4c_1x1)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4c_1x1_bn))

        (self.feed('inception_4b_output')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name=inception_4c_3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4c_3x3_reduce_bn)
             .conv(3, 3, 160, 1, 1, biased=False, relu=False, name=inception_4c_3x3)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4c_3x3_bn))

        (self.feed('inception_4b_output')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name=inception_4c_double3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4c_double3x3_reduce_bn)
             .conv(3, 3, 160, 1, 1, biased=False, relu=False, name=inception_4c_double3x3a)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4c_double3x3a_bn)
             .conv(3, 3, 160, 1, 1, biased=False, relu=False, name=inception_4c_double3x3b)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4c_double3x3b_bn))

        (self.feed('inception_4b_output')
             .avg_pool(3, 3, 1, 1, name=inception_4c_pool)
             .conv(1, 1, 96, 1, 1, biased=False, relu=False, name=inception_4c_pool_proj)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4c_pool_proj_bn))

        (self.feed('inception_4c_1x1_bn', 
                   'inception_4c_3x3_bn', 
                   'inception_4c_double3x3b_bn', 
                   'inception_4c_pool_proj_bn')
             .concat(3, name=inception_4c_output)
             .conv(1, 1, 96, 1, 1, biased=False, relu=False, name=inception_4d_1x1)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4d_1x1_bn))

        (self.feed('inception_4c_output')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name=inception_4d_3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4d_3x3_reduce_bn)
             .conv(3, 3, 192, 1, 1, biased=False, relu=False, name=inception_4d_3x3)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4d_3x3_bn))

        (self.feed('inception_4c_output')
             .conv(1, 1, 160, 1, 1, biased=False, relu=False, name=inception_4d_double3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4d_double3x3_reduce_bn)
             .conv(3, 3, 192, 1, 1, biased=False, relu=False, name=inception_4d_double3x3a)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4d_double3x3a_bn)
             .conv(3, 3, 192, 1, 1, biased=False, relu=False, name=inception_4d_double3x3b)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4d_double3x3b_bn))

        (self.feed('inception_4c_output')
             .avg_pool(3, 3, 1, 1, name=inception_4d_pool)
             .conv(1, 1, 96, 1, 1, biased=False, relu=False, name=inception_4d_pool_proj)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4d_pool_proj_bn))

        (self.feed('inception_4d_1x1_bn', 
                   'inception_4d_3x3_bn', 
                   'inception_4d_double3x3b_bn', 
                   'inception_4d_pool_proj_bn')
             .concat(3, name=inception_4d_output)
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name=inception_4e_3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4e_3x3_reduce_bn)
             .conv(3, 3, 192, 2, 2, biased=False, relu=False, name=inception_4e_3x3)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4e_3x3_bn))

        (self.feed('inception_4d_output')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name=inception_4e_double3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4e_double3x3_reduce_bn)
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name=inception_4e_double3x3a)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4e_double3x3a_bn)
             .conv(3, 3, 256, 2, 2, biased=False, relu=False, name=inception_4e_double3x3b)
             .batch_normalization(scale_offset=False, relu=True, name=inception_4e_double3x3b_bn))

        (self.feed('inception_4d_output')
             .max_pool(3, 3, 2, 2, name=inception_4e_pool))

        (self.feed('inception_4e_pool', 
                   'inception_4e_3x3_bn', 
                   'inception_4e_double3x3b_bn')
             .concat(3, name=inception_4e_output)
             .avg_pool(5, 5, 3, 3, padding=None, name=pool4_5x5_s3)
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name=loss2_conv)
             .batch_normalization(scale_offset=False, relu=True, name=loss2_conv_bn)
             .fc(1024, relu=False, name=loss2_fc)
             .batch_normalization(scale_offset=False, relu=True, name=loss2_fc_bn)
             .fc(1000, relu=False, name=loss2_classifier))

        (self.feed('inception_4e_output')
             .conv(1, 1, 352, 1, 1, biased=False, relu=False, name=inception_5a_1x1)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5a_1x1_bn))

        (self.feed('inception_4e_output')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name=inception_5a_3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5a_3x3_reduce_bn)
             .conv(3, 3, 320, 1, 1, biased=False, relu=False, name=inception_5a_3x3)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5a_3x3_bn))

        (self.feed('inception_4e_output')
             .conv(1, 1, 160, 1, 1, biased=False, relu=False, name=inception_5a_double3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5a_double3x3_reduce_bn)
             .conv(3, 3, 224, 1, 1, biased=False, relu=False, name=inception_5a_double3x3a)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5a_double3x3a_bn)
             .conv(3, 3, 224, 1, 1, biased=False, relu=False, name=inception_5a_double3x3b)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5a_double3x3b_bn))

        (self.feed('inception_4e_output')
             .avg_pool(3, 3, 1, 1, name=inception_5a_pool)
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name=inception_5a_pool_proj)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5a_pool_proj_bn))

        (self.feed('inception_5a_1x1_bn', 
                   'inception_5a_3x3_bn', 
                   'inception_5a_double3x3b_bn', 
                   'inception_5a_pool_proj_bn')
             .concat(3, name=inception_5a_output)
             .conv(1, 1, 352, 1, 1, biased=False, relu=False, name=inception_5b_1x1)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5b_1x1_bn))

        (self.feed('inception_5a_output')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name=inception_5b_3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5b_3x3_reduce_bn)
             .conv(3, 3, 320, 1, 1, biased=False, relu=False, name=inception_5b_3x3)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5b_3x3_bn))

        (self.feed('inception_5a_output')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name=inception_5b_double3x3_reduce)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5b_double3x3_reduce_bn)
             .conv(3, 3, 224, 1, 1, biased=False, relu=False, name=inception_5b_double3x3a)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5b_double3x3a_bn)
             .conv(3, 3, 224, 1, 1, biased=False, relu=False, name=inception_5b_double3x3b)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5b_double3x3b_bn))

        (self.feed('inception_5a_output')
             .max_pool(3, 3, 1, 1, name=inception_5b_pool)
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name=inception_5b_pool_proj)
             .batch_normalization(scale_offset=False, relu=True, name=inception_5b_pool_proj_bn))

        (self.feed('inception_5b_1x1_bn', 
                   'inception_5b_3x3_bn', 
                   'inception_5b_double3x3b_bn', 
                   'inception_5b_pool_proj_bn')
             .concat(3, name=inception_5b_output)
             .avg_pool(7, 7, 1, 1, padding='VALID', name=pool5_7x7_s1)
             .fc(180, relu=False, name=loss32_classifier)
             .softmax(name=softmax))