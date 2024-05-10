from model import DATASET_DIR, get_letter
import random
import string
import pygame
import sys
import cv2
import os
pygame.init()

WIDTH, HEIGHT = 32, 32
PIXEL_SIZE = 10
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

canvas = pygame.Surface((WIDTH * PIXEL_SIZE, HEIGHT * PIXEL_SIZE))
canvas.fill(WHITE)

screen = pygame.display.set_mode((WIDTH * PIXEL_SIZE, HEIGHT * PIXEL_SIZE))
pygame.display.set_caption("Hand Written Letter Classification")

drawing = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                mouse_pos = pygame.mouse.get_pos()
                canvas_x = mouse_pos[0] // PIXEL_SIZE
                canvas_y = mouse_pos[1] // PIXEL_SIZE

                pygame.draw.rect(canvas, BLACK, (canvas_x * PIXEL_SIZE, canvas_y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                canvas.fill(WHITE)
            elif event.key == pygame.K_RETURN:
                photo_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(15))+".png"
                pygame.image.save(canvas, photo_name)
                img = cv2.imread(photo_name)
                img = cv2.resize(img, (32, 32))
                (thresh, thresh_photo) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                cv2.imwrite(photo_name, thresh_photo)
                get_letter(photo_name)
                os.remove(photo_name)
                canvas.fill(WHITE)
    screen.blit(canvas, (0, 0))
    pygame.display.flip()