import Link from "next/link";
import { navLinks } from "../lib/data";

const Notion = ({ children }) => {
  return (
    <div className="flex flex-col items-center w-full notion">
      <nav className="w-[80%] flex justify-between py-10 text-xl">
        <Link href="/">Vincent Maladiere</Link>
        <div>
          {navLinks.map((l) => (
            <Link key={l.title} href={l.url} className="pl-5">
              {l.title}
            </Link>
          ))}
        </div>
      </nav>
      <div className="w-[80%] flex flex-col gap-10 pb-20">{children}</div>
    </div>
  );
};

export default Notion;
