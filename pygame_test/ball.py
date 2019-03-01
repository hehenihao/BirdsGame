import io, base64
import pygame
from pygame.locals import *

class Ball(pygame.sprite.Sprite):
    def __init__(self, color, initial_position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('src/ball.png').convert_alpha()
        self.rect = self.image.fill(color, None, BLEND_ADD)
        self.rect.topleft = initial_position

class MoveBall(Ball):
    def __init__(self,color,initial_position,speed,border):
        super(MoveBall,self).__init__(color,initial_position)
        self.speed = speed
        self.border = border
        self.update_time = 0

    def update(self,current_time):
        if self.update_time < current_time:
            if self.rect.left < 0 or self.rect.left > self.border[0]-self.rect.w:
                self.speed[0]*=-1
            if self.rect.top < 0 or self.rect.top > self.border[1]-self.rect.h:
                self.speed[1]*=-1
            self.rect.x += self.speed[0]
            self.rect.y += self.speed[1]
            self.update_time = current_time+10
        pass



pygame.init()
screen = pygame.display.set_mode([350, 350])

#ball = Ball((255, 0, 0), (100, 100))
balls = []
for color,location,speed in [([255, 0, 0], [50, 50],[2,3]),
                        ([0, 255, 0], [100, 100],[3,2]),
                        ([0, 0, 255], [150, 150],[4,3])]:
    balls.append(MoveBall(color,location,speed,[350,350]))

while True:
    if pygame.event.poll().type == QUIT:
        break
    screen.fill((0,0,0))
    current_time = pygame.time.get_ticks()
    for b in balls:
        b.update(current_time)
        screen.blit(b.image,b.rect)
    pygame.display.update()