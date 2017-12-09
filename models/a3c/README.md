# A3C

This implementation of A3C is based on the [OpenAI A3C implementation](https://github.com/openai/universe-starter-agent)
It uses a Tensorflow cluster to distribute the processing across multiple processes. Each process interacts by its own
with a Mancala Environment and asynchronously syncs with the weights of the parameter server.

## How to run the training

You need to run a command like the one below from the root directory of the project.

```bash
python models/a3c/train.py --num-workers 4 --log-dir /tmp/logs
```

After running this command, a tmux session with multiple windows will open where you could check the outputs of the
workers. To access the session type `tmux a` in the terminal. Then press `ctrl+b` followed by `w`. Then you will be able
to see all the open windows in this session.

On your browser visited: http://localhost:12345/ to see the Tensorboard and status of training

## Requirements
* install htop to monitor processes
* tmux version >= 1.7