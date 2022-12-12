'use strict';

const { merge } = require('webpack-merge');

const common = require('./webpack.common.js');

// Merge webpack configuration files
const config = (env, argv) =>
  merge(common, {
    devtool: argv.mode === 'production' ? false : 'source-map',
  });

module.exports = config;
