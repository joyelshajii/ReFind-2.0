interface CatMascotProps {
  className?: string;
  label?: string;
}

export default function CatMascot({
  className = "",
  label = "Black cat peeking over a gold line",
}: CatMascotProps) {
  return (
    <span
      aria-label={label}
      role="img"
      className={`cat-mascot ${className}`}
    />
  );
}
