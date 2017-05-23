import tensorflow as tf



# Counter Variable definition
with tf.name_scope('counter'):
    counter = tf.Variable(1, name="counter")
    tf.summary.scalar('counter', counter)

# Creation of a constant
two_op = tf.constant(2, name="const")

# Operations to perform in order to increment the variable value
new_value = tf.multiply(counter, two_op)
update = tf.assign(counter, new_value)

merged = tf.summary.merge_all()

# Initialize all variables
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # Increment the value of the variable in a session
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter("/path/to/logs", sess.graph)

    for i in range(5):
        summary, _ = sess.run([merged, update])
        summary_writer.add_summary(summary, i)
        print(sess.run(counter))

tensorboard --logdir="/path/to/logs"