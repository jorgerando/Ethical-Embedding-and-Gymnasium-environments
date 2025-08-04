from gymnasium_envs.PublicCivilityDependencis.Environment import *

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import pygame
from pygame_emojis import load_emoji

import time
import sys

class PublicCivilityGame(gym.Env):

    def __init__(self,WS=[1.0,0.71]):

        NB_ACTIONS = 6
        self.WS = WS
        self.AC = 0
        self.IC = 1
        self.GC = 2
        self.map_width = 4
        self.map_height = 6

        self.env = Environment()
        self.action_space = gym.spaces.Discrete(NB_ACTIONS) # 5 acciones
        self.observation_space = gym.spaces.Box(0,self.map_height*self.map_width,shape=(3,),dtype = int) # un vecto de 3 numero , se pasa el mapa de 2d a 1d para simplificar
        #print(self.observation_space.sample())
        self.max_steps = 20
        self.steps = 0

        #render
        self.window = None
        self.windows_size = 300

    def setWeights(self,WS):
        self.WS = WS

    def reset(self):

        self.env.hard_reset()
        self.steps = 0
        st = self.env.get_state()
        return tuple(st), {}

    def step(self,action):

        new_state, reward, dones = self.env.step([action])
        self.steps += 1

        WE = self.WS

        etical_renward = self.WS[0] * reward[0] +  self.WS[1] * reward[1] # indv renward + sum etical renwards

        info = {'Individual': reward[0] ,'Etical': reward[1]}

        return tuple(new_state) , etical_renward , dones[0] or self.steps >= self.max_steps , False , info

    def render(self):

        if self.window is None :
         pygame.init()
         pygame.display.init()
         self.window = pygame.display.set_mode((self.map_width*self.windows_size ,self.map_height*self.windows_size ))
         my_font = pygame.font.SysFont('Comic Sans MS', 30)
         pygame.display.set_caption("Public Civility Game 🤖 🏢 👱")
         self.clock = pygame.time.Clock()

        self.window.fill((255,255,255))

        for event in pygame.event.get():
         if event.type == pygame.QUIT:
          pygame.quit()
          sys.exit()

        for y in range(6):
            for x in range(4):

                title = self.env.map_tileset[y][x]

                size = (self.windows_size,self.windows_size)

                if title == self.AC :
                   surface = load_emoji('⬜',size )
                if title == self.IC :
                     surface = load_emoji('⬛', size)
                if title == self.GC :
                    surface = load_emoji('🗑', size)

                self.window.blit(surface, (x*self.windows_size,y*self.windows_size))

                size = (self.windows_size,self.windows_size)

                if [y,x] == self.env.agents[1].get_position() :
                    surface = load_emoji('🤖', size)

                self.window.blit(surface, (x*self.windows_size,y*self.windows_size))

                if [y,x] == self.env.agents[0].get_position() :
                    surface = load_emoji('👱', size)

                self.window.blit(surface, (x*self.windows_size,y*self.windows_size))

                if [y,x] == self.env.items[0].get_position():
                    surface = load_emoji('', size)

                self.window.blit(surface, (x*self.windows_size,y*self.windows_size))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
     if self.window is not None:
        pygame.display.quit()
        pygame.quit()

'''
env = PublicCivilityGame()
# This will catch many common issues
try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
'''
