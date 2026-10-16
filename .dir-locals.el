;;; Directory Local Variables            -*- no-byte-compile: t -*-
;;; For more information see (info "(emacs) Directory Variables")

;; Activates the project's uv-managed .venv, wherever the repository lives.
((python-mode . ((eval . (when (fboundp 'pyvenv-activate)
                           (pyvenv-activate
                            (expand-file-name ".venv" (locate-dominating-file
                                                       default-directory ".dir-locals.el")))))))))
