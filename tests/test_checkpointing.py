from src.lit.checkpointing import last_checkpoint


def test_last_checkpoint_is_independent_of_top_k_metric(tmp_path):
    callback = last_checkpoint(tmp_path)

    assert callback.monitor is None
    assert callback.filename == "last"
    assert callback.save_top_k == 1
    assert callback.save_last is False
    assert callback.save_on_exception is True
    assert callback._every_n_epochs == 1
    assert callback._save_on_train_epoch_end is True
    assert callback._enable_version_counter is False
