/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,jsx,ts,tsx,md,mdx}",
    "./components/**/*.{js,jsx,ts,tsx,md,mdx}",

    // Or if using `src` directory:
    "./src/**/*.{js,jsx,ts,tsx,md,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#F8F8F8",
        lightPurple: "#E2E0FF",
      },
    },
  },
  plugins: [],
};
