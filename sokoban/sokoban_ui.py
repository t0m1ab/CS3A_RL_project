import numpy as np
import gradio as gr
import gymnasium as gym
import argparse

from sokoban_env import SokobanEnv
from dataloaders import SokobanDataLoader, DeepMindBoxobanLoader, MySokobanLoader

WALL_COLOR = (176, 61, 0)
BOX_COLOR = (215, 196, 0)


class Interface():

    DICT_CSS = {
        "push-button": {
            "color": "white",
            "background": f"rgb{str(BOX_COLOR).replace(' ','')}",
        },
        "move-button": {
            "color": "white",
            "background": f"rgb{str(WALL_COLOR).replace(' ','')}",
        },
        "push-label": {
            "color": "black",
            "background": f"rgb{str(BOX_COLOR).replace(' ','')}",
        },
        "move-label": {
            "color": "black",
            "background": f"rgb{str(WALL_COLOR).replace(' ','')}",
        },
    }

    def __init__(
            self,
            env_name: str = None,
            map_txt: str = None,
            map_collection: SokobanDataLoader = None,
            env_kwargs: dict = None,
        ) -> None:

        self.str_wall_color = str(WALL_COLOR).replace(" ", "")
        self.css = self.__build_css()
        self.logs = ""

        self.env = gym.make(
            id=env_name,
            map_txt=map_txt,
            map_collection=map_collection,
            **env_kwargs,
        )

        self.env.reset()
    
    def __build_css(self, dict_css: dict = None) -> str:
        """ Build the CSS string from a dictionary of CSS. If dict_css is None, use DICT_CSS attribute. """
        dict_css = Interface.DICT_CSS if dict_css is None else dict_css
        css = ""
        for key, value in dict_css.items():
            css += f".{key} {{"
            for k, v in value.items():
                css += f"{k}:{v};"
            css = css[:-1] + "} "
        return css
    
    def __action_button(self, action: int) -> tuple[np.ndarray, str]:
        """ Perform the action and return the new image of the environment. """
        (env_map, player_position), reward, terminated, truncated, info = self.env.step(action)
        self.__add_logs(f"#{self.env.unwrapped.num_env_steps} - {info['action_result']} | Reward = {reward}")
        return self.env.unwrapped.get_image(), self.logs
    
    def __reset_episode(self) -> tuple[np.ndarray, str]:
        self.env.unwrapped.reset_episode()
        self.logs = ""
        return self.env.unwrapped.get_image(), self.logs
    
    def __reset(self) -> tuple[np.ndarray, str]:
        self.env.reset()
        self.logs = ""
        return self.env.unwrapped.get_image(), self.logs

    def __add_logs(self, txt_line: str = None) -> None:
        """ Add logs to the textbox. """
        if txt_line is None:
            self.logs += "\nNothing to say..."
        else:
            self.logs += f"\n{txt_line}" if len(self.logs) > 0 else txt_line
    
    def __clear_logs(self) -> str:
        """ Clear the logs. """
        self.logs = ""
        return self.logs

    def launch(self, share: bool = False) -> None:

        with gr.Blocks(css=self.css) as page:

            gr.HTML(value=f"<h1 style=color:rgb{self.str_wall_color};font-size:4em>SOKOBAN</h1>")

            with gr.Row():

                with gr.Column(scale=1.0):

                    image = gr.Image(value=self.env.unwrapped.get_image())

                    with gr.Row():
                        
                        with gr.Row(): # PUSH buttons
                            label_push_btn = gr.Button(value="PUSH", elem_classes="push-label", interactive=False)
                            btn_push_up = gr.Button(value="UP", elem_classes="push-button")
                            btn_push_down = gr.Button(value="DOWN", elem_classes="push-button")
                            btn_push_left = gr.Button(value="LEFT", elem_classes="push-button")
                            btn_push_right = gr.Button(value="RIGHT", elem_classes="push-button")
                        
                        with gr.Row(): # MOVE buttons
                            label_move_btn = gr.Button(value="MOVE", elem_classes="move-label", interactive=False)
                            btn_move_up = gr.Button(value="UP", elem_classes="move-button", interactive=not self.env.unwrapped.merge_move_push)
                            btn_move_down = gr.Button(value="DOWN", elem_classes="move-button", interactive=not self.env.unwrapped.merge_move_push)
                            btn_move_left = gr.Button(value="LEFT", elem_classes="move-button", interactive=not self.env.unwrapped.merge_move_push)
                            btn_move_right = gr.Button(value="RIGHT", elem_classes="move-button", interactive=not self.env.unwrapped.merge_move_push)
                
                with gr.Column(scale=2.0):

                    with gr.Row():

                        # RESET EPISODE button
                        btn_reset_episode = gr.Button(value="Reset episode")

                        # RESET button
                        btn_reset = gr.Button(value="Reset")

                        # EXTRA button
                        btn_clear = gr.Button(value="Clear")

                    with gr.Row():

                        # LOGS
                        env_logs = gr.Textbox(
                            value=self.logs,
                            label="History",
                            placeholder="Logs of actions will be displayed here...",
                            lines=1,
                            interactive=False)
                    
                    # PUSH buttons actions
                    btn_push_up.click(lambda: self.__action_button(0), outputs=[image, env_logs])
                    btn_push_down.click(lambda: self.__action_button(1), outputs=[image, env_logs])
                    btn_push_left.click(lambda: self.__action_button(2), outputs=[image, env_logs])
                    btn_push_right.click(lambda: self.__action_button(3), outputs=[image, env_logs])

                    # MOVE buttons actions
                    btn_move_up.click(lambda: self.__action_button(4), outputs=[image, env_logs])
                    btn_move_down.click(lambda: self.__action_button(5), outputs=[image, env_logs])
                    btn_move_left.click(lambda: self.__action_button(6), outputs=[image, env_logs])
                    btn_move_right.click(lambda: self.__action_button(7), outputs=[image, env_logs])
                    
                    # SETTINGS buttons
                    btn_reset_episode.click(self.__reset_episode, outputs=[image, env_logs])
                    btn_reset.click(self.__reset, outputs=[image, env_logs])
                    btn_clear.click(self.__clear_logs, outputs=[env_logs])
        
        page.launch(share=share)


if __name__ == "__main__":

    ## Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--merge_move_push",
        action="store_true",
        default=False,
        help="If true, merge move and push actions",
    )
    parser.add_argument(
        "-r",
        "--reset_mode",
        type=str,
        choices=["random", "next"],
        default="random",
        help="Specify the reset mode and must be in {random,next}",
    )
    args = parser.parse_args()
    env_kwargs = {
        "merge_move_push": args.merge_move_push, 
        "reset_mode": args.reset_mode,
    }

    ## Register the Sokoban environment
    namespace = "sokoban"
    env_id = "sokoban-v0"
    with gym.envs.registration.namespace(ns=namespace):
        gym.register(id=env_id, entry_point=SokobanEnv)
    
    ## Load a Sokoban map collection
    sokoban_data = MySokobanLoader(level="easy", file_id=0)
    # sokoban_data = DeepMindBoxobanLoader(level="medium", file_id=0)

    ## Create the UI
    game = Interface(
        env_name=f"{namespace}/{env_id}",
        map_collection=sokoban_data,
        env_kwargs=env_kwargs,
    )

    ## LAUNCH
    game.launch(share=False)