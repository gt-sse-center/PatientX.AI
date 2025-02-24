# Tmux: Why Use It and How to Start a Session

## Why Use Tmux?
[Tmux (Terminal Multiplexer)](https://github.com/tmux/tmux/wiki) is a powerful tool that allows you to:

- **Manage Multiple Terminal Sessions:** Run multiple shell sessions within a single window and easily switch between them.
- **Persistent Sessions:** Keep your sessions running even after disconnecting, which is particularly useful for SSH connections. **This can be particularly useful here since running the full PatientX.AI may take some time depending on the size of the dataset and model configuration. Without `tmux`, if the ssh connection disconnects, the code may not run to completion**
- **Efficient Workflow:** Split your terminal into multiple panes, allowing you to work on different tasks simultaneously.

## Basic Tmux Commands

### Start a New Session
```bash
tmux new -s session_name
```
This creates a new session named `session_name`.

### Detach from a Session
To detach from a session without stopping it, press:
```
Ctrl + b, then d
```

### List all Sessions
To list all running sessions
```
tmux ls
```

### Reattach to an Existing Session
To reattach to an existing session
```
tmux attach -t session_name
```

## Learn More

For more details and advanced usage, refer to the [official tmux documentation](https://github.com/tmux/tmux/wiki)
