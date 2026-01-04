export default function Footer() {
  return (
    <footer className="footer">
      <div className="container footer-content">
        <p className="footer-text">
          © {new Date().getFullYear()} EcoSight AI — Presidential AI Challenge
        </p>
        <p className="footer-text">
          Built with satellite imagery & machine learning
        </p>
      </div>
    </footer>
  )
}


