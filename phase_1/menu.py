import os
import sys
import subprocess
import pygame

pygame.init()

WIDTH = 600
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fitonexa")
FPS = 60
clock = pygame.time.Clock()

font = pygame.font.Font("freesansbold.ttf", 24)

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name):
    script_path = os.path.join(SCRIPT_FOLDER, script_name)
    if os.path.exists(script_path):
        subprocess.run([sys.executable, script_path])
    else:
        print(f"Script not found: {script_path}")


def draw_button(rect, text):
    pygame.draw.rect(screen, "light gray", rect, 0, 20)
    pygame.draw.rect(screen, "dark gray", rect, 5, 20)
    text_surface = font.render(text, True, "black")
    screen.blit(text_surface, (rect.x + 12, rect.y + 8))


button_defs = [
    {
        "rect": pygame.Rect(150, 170, 340, 40),
        "text": "1. Squats Analyzer",
        "action": lambda: run_script("squat_analyzer.py"),
    },
    {
        "rect": pygame.Rect(150, 230, 340, 40),
        "text": "2. Lunges Analyzer",
        "action": lambda: run_script("lunges_analyzer.py"),
    },
    {
        "rect": pygame.Rect(150, 290, 340, 40),
        "text": "3. Bicep Curl Analyzer",
        "action": lambda: run_script("bicep_curl_analyzer.py"),
    },
    {
        "rect": pygame.Rect(150, 350, 340, 40),
        "text": "4. Pull Ups Analyzer",
        "action": lambda: print("Pull Ups Analyzer is not yet implemented."),
    },
    {
        "rect": pygame.Rect(150, 410, 340, 40),
        "text": "5. High Knees Analyzer",
        "action": lambda: run_script("high_knees_analyzer.py"),
    },
    {
        "rect": pygame.Rect(150, 470, 340, 40),
        "text": "6. Lateral Raises Analyzer",
        "action": lambda: run_script("lateral_raises_analyzer.py"),
    },
]

exit_rect = pygame.Rect(470, 530, 80, 40)

run = True
while run:
    screen.fill("light blue")
    clock.tick(FPS)

    screen.blit(font.render("Welcome to PowerPulse!", True, "black"), (150, 60))
    screen.blit(
        font.render("Choose the exercise you want to perform", True, "black"),
        (50, 130),
    )

    for button in button_defs:
        draw_button(button["rect"], button["text"])

    pygame.draw.rect(screen, "light gray", exit_rect, 0, 20)
    pygame.draw.rect(screen, "dark gray", exit_rect, 5, 20)
    exit_text = font.render("Exit", True, "black")
    screen.blit(exit_text, (exit_rect.x + 15, exit_rect.y + 8))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            for button in button_defs:
                if button["rect"].collidepoint(mouse_pos):
                    button["action"]()
                    break
            if exit_rect.collidepoint(mouse_pos):
                run = False

    pygame.display.flip()

pygame.quit()
sys.exit()






