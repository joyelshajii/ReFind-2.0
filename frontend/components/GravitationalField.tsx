"use client";

import { useEffect, useRef } from "react";

interface Star {
  x: number;
  y: number;
  depth: number;
  radius: number;
  opacity: number;
  phase: number;
}

export default function GravitationalField() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
    const pointer = { x: 0, y: 0, targetX: 0, targetY: 0 };
    let width = window.innerWidth;
    let height = window.innerHeight;
    let stars: Star[] = [];
    let animationFrameId = 0;
    let lastTime = 0;
    let elapsed = 0;

    const draw = (time: number) => {
      const reducedMotion = motionPreference.matches;
      const delta = lastTime ? Math.min(time - lastTime, 50) : 16;
      lastTime = time;
      if (!reducedMotion) elapsed += delta;

      // Ease toward the cursor; nearer dots travel farther in the same direction.
      const easing = 1 - Math.exp(-delta / 150);
      pointer.x = reducedMotion ? 0 : pointer.x + (pointer.targetX - pointer.x) * easing;
      pointer.y = reducedMotion ? 0 : pointer.y + (pointer.targetY - pointer.y) * easing;

      ctx.clearRect(0, 0, width, height);
      for (const star of stars) {
        const drift = reducedMotion ? 0 : Math.sin(elapsed / 8000 + star.phase) * 3;
        const x = star.x * (width + 120) - 60 + pointer.x * star.depth + drift;
        const y = star.y * (height + 120) - 60 + pointer.y * star.depth + drift * 0.6;

        ctx.beginPath();
        ctx.arc(x, y, star.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${star.opacity})`;
        ctx.fill();

        if (star.depth > 0.85) {
          ctx.beginPath();
          ctx.arc(x, y, star.radius * 2.8, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 255, 255, ${star.opacity * 0.08})`;
          ctx.fill();
        }
      }

      // Keep background dots clear of the homepage search bar and its suggestions.
      const searchArea = document
        .querySelector(".refind-home-layout .refind-search-bar")
        ?.getBoundingClientRect();
      if (searchArea) {
        ctx.clearRect(
          searchArea.left - 64,
          searchArea.top - 56,
          searchArea.width + 128,
          searchArea.height + 112,
        );
      }

      if (!reducedMotion && !document.hidden) {
        animationFrameId = requestAnimationFrame(draw);
      }
    };

    const restart = () => {
      cancelAnimationFrame(animationFrameId);
      lastTime = 0;
      draw(performance.now());
    };

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

      const count = width < 640 ? 70 : width < 1024 ? 120 : 180;
      if (stars.length !== count) {
        stars = Array.from({ length: count }, () => {
          const depth = 0.2 + Math.random() * 0.8;
          return {
            x: Math.random(),
            y: Math.random(),
            depth,
            radius: 0.45 + depth * 1.05,
            opacity: 0.16 + depth * 0.5,
            phase: Math.random() * Math.PI * 2,
          };
        });
      }
      restart();
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (event.pointerType === "touch" || motionPreference.matches) return;
      pointer.targetX = (event.clientX / width - 0.5) * 100;
      pointer.targetY = (event.clientY / height - 0.5) * 76;
    };

    const resetPointer = () => {
      pointer.targetX = 0;
      pointer.targetY = 0;
    };

    const handleScroll = () => {
      if (motionPreference.matches) restart();
    };

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", handlePointerMove, { passive: true });
    window.addEventListener("blur", resetPointer);
    window.addEventListener("scroll", handleScroll, { passive: true });
    document.documentElement.addEventListener("pointerleave", resetPointer);
    document.addEventListener("visibilitychange", restart);
    motionPreference.addEventListener("change", restart);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("blur", resetPointer);
      window.removeEventListener("scroll", handleScroll);
      document.documentElement.removeEventListener("pointerleave", resetPointer);
      document.removeEventListener("visibilitychange", restart);
      motionPreference.removeEventListener("change", restart);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="fixed inset-0 pointer-events-none z-0 block w-full h-full"
    />
  );
}
