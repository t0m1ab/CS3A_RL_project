import os
import gradio as gr
import gymnasium as gym

from sokoban_env import SokobanEnv
from dataloaders import SokobanDataLoader, DeepMindBoxobanLoader, MySokobanLoader


class Interface():

    def __init__(
            self,
            env_name: str = None,
            map_txt: str = None,
            map_collection: SokobanDataLoader = None,
        ) -> None:

        self.env = gym.make(
            id=env_name,
            map_txt=map_txt,
            map_collection=map_collection,
        )

        self.env.reset()

        assert self.env.unwrapped.num_env_steps == 0
    
    def action_button(self, action: int) -> None:
        """ Perform the action and return the new image of the environment. """
        self.env.step(action)
        return self.env.unwrapped.get_image()

    def launch(self, share: bool = False) -> None:

        with gr.Blocks() as page:

            # show an image on the left and buttons on the right
            with gr.Row():
                image = gr.Image(value=self.env.unwrapped.get_image())
                with gr.Column():
                    btn_up = gr.Button(value="Up")
                    btn_down = gr.Button(value="Down")
                    btn_left = gr.Button(value="Left")
                    btn_right = gr.Button(value="Right")
                    btn_up.click(lambda: self.action_button(0), outputs=[image])
                    btn_down.click(lambda: self.action_button(1), outputs=[image])
                    btn_left.click(lambda: self.action_button(2), outputs=[image])
                    btn_right.click(lambda: self.action_button(3), outputs=[image])
        
        page.launch(share=share)


if __name__ == "__main__":

    # register the environment
    namespace = "sokoban"
    env_id = "sokoban-v0"
    with gym.envs.registration.namespace(ns=namespace):
        gym.register(id=env_id, entry_point=SokobanEnv)
    
    # load a map collection
    # sokoban_data = MySokobanLoader(level="easy", file_id=0)
    sokoban_data = DeepMindBoxobanLoader(level="medium", file_id=0)

    # create the interface
    game = Interface(
        env_name=f"{namespace}/{env_id}",
        map_collection=sokoban_data,
    )

    # launch the interface
    game.launch(share=False)

