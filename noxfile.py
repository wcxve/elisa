import nox

PYTHON_VERSIONS = ['3.9', '3.10', '3.11']


@nox.session(python=PYTHON_VERSIONS)
def test(session: nox.Session) -> None:
    session.install('.[test]')
    session.run('pytest', '--cov-report=xml', '--cov=elisa', *session.posargs)
