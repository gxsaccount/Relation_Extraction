import tensorflow as tf

a=tf.constant([
    [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]],
    [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]]
               ])
b=tf.reshape(a,[2,3*6])
sess = tf.Session()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(b))

