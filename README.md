# deepface
check_point = keras.callbacks.ModelCheckpoint(model_savedir + '/'+'{epoch:06d}-{acc:.8f}-{val_acc:.8f}.hdf5', monitor='acc',
								verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
