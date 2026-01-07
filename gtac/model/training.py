import os
import copy
import numpy as np
import tensorflow as tf
import tf_keras as keras
import tracemalloc
from gtac.utils import MPDataset
# Import necessary functions
from gtac.tensorflow_transformer import masked_loss, masked_accuracy

def train(self,
          train_data_dir,
          ckpt_save_path=None,
          validation_split=0.1,
          epochs=1,
          initial_epoch=0,
          batch_size=4,
          profile=True,
          distributed=False,
          latest_ckpt_only=False,
          log_dir='tensorboard',
          excluded_files: list = None,
          freeze_layers=False,
          shuffle_seed: int = 0
          ):
    train_data_dir = train_data_dir + ("/" if train_data_dir[-1] != "/" else "")
    
    if ckpt_save_path is None:
        print("WARNING: ckpt_save_path is not specified, the trained model will not be saved during training!")
    else:
        ckpt_save_path = ckpt_save_path + ("/" if ckpt_save_path[-1] != "/" else "")

        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)

    if freeze_layers:
        self.freeze_layers(freeze_encoder=True)

    train_files = os.listdir(train_data_dir)
    print("%d training files listed" % len(train_files))
    train_files.sort()
    print("training files sorted")
    np.random.seed(shuffle_seed)
    np.random.shuffle(train_files)
    print("training files shuffled")

    self._transformer.return_cache = False

    if excluded_files is not None:
        print("excluded files is not None, filtering training files...")
        excluded_files = set(excluded_files)
        new_train_files = []
        for file in train_files:
            if file not in excluded_files:
                new_train_files.append(file)
        print("training files filtered, from %d to %d" % (len(train_files), len(new_train_files)))
        train_files = new_train_files

    train_files = [(train_data_dir + file) for file in train_files]
    self_copied = copy.copy(self)
    self_copied._transformer = None
    self_copied._transformer_inference = None

    mp_dataset = MPDataset(train_files, self_copied.load_and_encode_formatted, validation_split=validation_split, num_processes=8)

    output_signature = (
        {
            'inputs': tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32),
            'enc_pos_encoding': tf.TensorSpec(shape=(self.max_seq_length, self.max_tree_depth * 2),
                                              dtype=tf.float32),
            'targets': tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32), #######################################输入了target
            'dec_pos_encoding': tf.TensorSpec(shape=(self.max_seq_length, self.max_tree_depth * 2),
                                              dtype=tf.float32),
            'enc_action_mask': tf.TensorSpec(shape=(self.max_seq_length, self.vocab_size),
                                             dtype=tf.bool),
            'dec_action_mask': tf.TensorSpec(shape=(self.max_seq_length, self.vocab_size),
                                              dtype=tf.bool)
        }, tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32)
    )
    print("Creating TensorFlow dataset...")
    train_dataset = tf.data.Dataset.from_generator(mp_dataset.train_generator,
                                                   output_signature=output_signature) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = tf.data.Dataset.from_generator(mp_dataset.validation_generator,
                                                        output_signature=output_signature) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("Dataset creation completed")
    
    if profile:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        summary_writer = tf.summary.create_file_writer(log_dir)

    if distributed:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            transformer = self._get_tf_transformer()
            if self.ckpt_path is not None:
                transformer.load_weights(self.ckpt_path)
            # learning_rate = CustomSchedule(self.embedding_width)
            optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            # optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            
            transformer.compile(
                optimizer=optimizer,
                loss=masked_loss,
                metrics=[masked_accuracy]
            )
    else:
        transformer = self._transformer
        # learning_rate = CustomSchedule(self.embedding_width)
        optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9) # for pretraining
        # optimizer = keras.optimizers.Adam(learning_rate=1e-6, beta_1=0.9, beta_2=0.98, epsilon=1e-9) # for finetuning
        # optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        transformer.compile(
            optimizer=optimizer,
            loss=masked_loss,
            metrics=[masked_accuracy],
        )

    class LogCallback(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            print(f"Batch {batch} finished with logs: {logs}")
            # step = transformer.optimizer.iterations.numpy()
            # with summary_writer.as_default():
            #     # Write loss and accuracy
            #     tf.summary.scalar('loss', logs['loss'], step=batch)
            #     tf.summary.scalar('accuracy', logs['accuracy'], step=batch)
            # pass

        def on_epoch_end(self, epoch, logs=None):
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            for stat in top_stats:
                print(stat)

    log = LogCallback()

    callbacks = []
    if ckpt_save_path is not None:
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=ckpt_save_path + 'model-{epoch:04d}',
            save_weights_only=True,
            save_freq=(len(mp_dataset) * (epochs - initial_epoch) // batch_size) if latest_ckpt_only else 'epoch') # type: ignore
        callbacks.append(checkpoint)

    print("Starting training, preparing to call fit() method")
    transformer.fit(train_dataset,
                    initial_epoch=initial_epoch,
                    epochs=epochs,
                    validation_data=validation_dataset,
                    callbacks=callbacks,
                    verbose=1)
    mp_dataset.process.terminate()
    print("training finished")

    if profile:
        # Simple profiling without using trace_export
        print("Profiling completed. Check log directory for TensorBoard logs.")

    self._transformer.return_cache = True 