from IPython.terminal.embed import InteractiveShellEmbed
from inspect import currentframe
from traitlets.config.loader import Config

cfg = Config()
# prompt_config = cfg.PromptManager
# prompt_config.in_template = 'N.In <\\#>: '
# prompt_config.in2_template = '   .\\D.: '
# prompt_config.out_template = 'N.Out<\\#>: '

# Messages displayed when I drop into and exit the shell.
banner_msg = ("\n**Nested Interpreter:\n"
"Hit Ctrl-D to exit interpreter and continue program.\n"
"Note that if you use %kill_embedded, you can fully deactivate\n"
"This embedded instance so it will never turn on again")   
exit_msg = '**Leaving Nested interpreter'
ipshell = InteractiveShellEmbed(config=cfg, banner1=banner_msg, exit_msg=exit_msg)

def ipsh():
    frame = currentframe().f_back
    msg = 'Stopped at {0.f_code.co_filename} and line {0.f_lineno}'.format(frame)
    ipshell(msg,stack_depth=2) # Go back one level!
