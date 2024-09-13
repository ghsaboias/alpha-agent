# Claude-Integrated Discord Bot

This project contains a Discord bot that interfaces with Claude AI, allowing for natural language interactions within a specified Discord server.

## Project structure

Below is an overview of the simplified project structure:

```
├── .env              -> environment variables (keep this private!)
├── app.js            -> main entrypoint for the bot
├── claude-api.js     -> Claude AI integration logic
├── config.json       -> basic configuration for the bot
├── package.json      -> project dependencies and scripts
└── .gitignore        -> specifies intentionally untracked files
```

## Running the bot locally

Before you start, you'll need to:

1. Install [NodeJS](https://nodejs.org/en/download/)
2. [Create a Discord app](https://discord.com/developers/applications) with the proper permissions:
   - `bot` (with Send Messages enabled)
3. Obtain an API key from Anthropic for Claude AI access

### Setup project

Clone the project and install dependencies:

```
git clone [your-repository-url]
cd [your-project-directory]
npm install
```

### Configure the bot

Create a `.env` file in the project root and add the following:

```
APP_ID=your_discord_app_id
DISCORD_TOKEN=your_discord_bot_token
CLAUDE_API_KEY=your_claude_api_key
GUILD_ID=your_discord_server_id
AUTHORIZED_USER_ID=your_discord_user_id
```

Replace the placeholder values with your actual credentials and IDs.

### Run the bot

After configuration, run the bot:

```
npm start
```

## Usage

Once the bot is running and invited to your server, simply send a message in the specified server. The bot will process messages from the authorized user and respond with Claude's replies.

## Customization

To modify the bot's behavior or add new features, edit the `app.js` and `claude-api.js` files. The `config.json` file can be used for any additional configuration options you might want to add.

## Security Note

Keep your `.env` file and all API keys private. Never commit them to version control or share them publicly.

## Support

For questions about the Discord API, join the [Discord Developers server](https://discord.gg/discord-developers).

For Claude AI-related queries, refer to the [Anthropic documentation](https://www.anthropic.com).
