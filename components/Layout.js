import Link from "next/link";
import { navLinks } from "../lib/data";

const Layout = ({ children }) => {
  return (
    <div className="flex flex-col items-center w-full layout">
      <nav className="w-[90%] md:w-[80%] flex flex-wrap justify-between py-10 text-xl">
        <Link href="/">
          Vincent Maladiere
        </Link>
        
        <div className="flex flex-wrap">
          {navLinks.map((l) => (
            <Link key={l.title} href={l.url} className="pl-0 md:pl-5 pr-5 md:pr-0">
              {l.title}
            </Link>
          ))}
        </div>
      </nav>

      <div className="w-[90%] md:w-[80%] flex flex-col items-center gap-10">
        {children}
      </div>
    </div>
  );
};

export default Layout;
