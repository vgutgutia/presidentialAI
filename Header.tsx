import { Link, NavLink } from 'react-router-dom'

export default function Header() {
  return (
    <header className="header">
      <div className="container header-inner">
        <Link to="/" className="logo">
          <div className="logo-icon">
            {/* TODO: Add icon/SVG */}
            🛰
          </div>
          <span>EcoSight AI</span>
        </Link>

        <nav className="nav">
          <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            Home
          </NavLink>
          <NavLink to="/demo" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            Live Demo
          </NavLink>
          <a 
            href="https://github.com" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="btn btn-secondary btn-sm"
          >
            GitHub
          </a>
        </nav>
      </div>
    </header>
  )
}


