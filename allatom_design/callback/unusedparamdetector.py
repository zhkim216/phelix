from lightning.pytorch.callbacks import Callback

class UnusedParamDetector(Callback):
    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step < 5 and trainer.is_global_zero:
            unused = []
            for name, param in pl_module.named_parameters():
                if param.requires_grad and (param.grad is None):
                    unused.append(name)
            if unused:
                print("\n[UnusedParamDetector] UNUSED PARAMS (no grad at step 0):")
                for n in unused:
                    print(f" - {n}")
                print("")