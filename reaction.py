import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Identifiants (pour tests, écrivez-les directement ou gérez-les via variables d'environnement)
EMAIL_ADDRESS = "EMAIL"
EMAIL_PASSWORD = "PASSWORD"

def envoyer_email(destinataire, objet, message_html, message_text=None):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = objet
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = destinataire

    if message_text:
        msg.attach(MIMEText(message_text, 'plain'))
    msg.attach(MIMEText(message_html, 'html'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, destinataire, msg.as_string())
        print(f"E-mail envoyé avec succès à {destinataire}")
    except smtplib.SMTPAuthenticationError as auth_err:
        print("Échec de l'authentification SMTP :", auth_err.smtp_code, auth_err.smtp_error)
        print("→ Vérifiez votre adresse, votre mot de passe d’application et que l’authentification à deux facteurs est activée.")
    except Exception as e:
        print("Échec de l'envoi :", e)

if __name__ == '__main__':
    envoyer_email(
        'EMAIL_DESTINATAIRE',
        "Test d'envoi avec Python via Gmail",
        "<h1>Bonjour</h1><p>Ceci est un <b>test</b> d'envoi d'e-mail via <i>Python</i>.</p>",

    )