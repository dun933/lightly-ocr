package db

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	log "github.com/sirupsen/logrus"

	// mysql driver
	_ "github.com/go-sql-driver/mysql"
)

//constant
// const userQuery string = "CREATE TABLE IF NOT EXISTS database (userName varchar(255), userScore int, imgPath varchar(255));"
// const co2Query string = "CREATE TABLE IF NOT EXISTS co2 (items varchar(255), emission float64);"
const dbName string = "backend-app"
const devEnv = false
const ifExists = false

// connected checks whether the database is connected or not
var connected bool

// PingTimeout measures how long we should ping when attempting to reconnect to the database
var PingTimeout time.Duration = 1 * time.Second

// SleepTimeout returns the amount of time to wait during the goroutine when reconnecting
var SleepTimeout time.Duration = 5 * time.Second

// User contains the general table to store username and given score for the item
type User struct {
	userName  string
	userScore float64
	imgPath   string
}

// CO2 contains data about item with agrigation sum of greenhouse gasses (CO2, NH4, etc)
type CO2 struct {
	items    string
	emission float64
}

type DB struct {
	Client *sql.DB
	Driver string
	URL    string
}

// Ping wraps sql.Ping
func (db *DB) Ping() error {
	return db.Client.Ping()
}

// PingContext wraps around sql.PingContext
func (db *DB) PingContext(ctx context.Context) error {
	return db.Client.PingContext(ctx)
}

// Exec is just sql.Exec wrapper
func (db *DB) Exec(sql string, params ...interface{}) (sql.Result, error) {
	res, err := db.Client.Exec(sql, params...)
	return res, err
}

// Query wraps sql.Query
func (db *DB) Query(sql string, params ...interface{}) (*sql.Rows, error) {
	res, err := db.Client.Query(sql, params...)
	return res, err
}

func (db *DB) CreateTable(st interface{}, ifExists bool) error {
	t, err := NewTable(st)
	if err != nil {
		return err
	}

	_, err := db.Exec(NewTableQuery(t.SQLName, t.SQLOptions(), ifExists))
	return err
}

// Connect responds for connecting the database
func Connect(driver, url string) (*DB, error) {
	client, err := sql.Open(driver, url)
	if err != nil {
		return nil, err
	}
	return &DB{
		Client: client,
		Driver: driver,
		URL:    url,
	}, nil
}

// ConnectURL returns connect string (static atm)
// TODO: added options for users/password, etc
func ConnectURL(devEnv bool) string {
	var url string
	if devEnv {
		url = fmt.Sprintf("root:toor@tcp(localhost)/%s", dbName)
	} else {
		url = fmt.Sprintf("application:application123@tcp(localhost)/%s", dbName)
	}
	return url
}

// init tries to reconnect to database when there is no connection established
func init() {
	connected = false
	db, _ := Connect("mysql", ConnectURL(devEnv))

	err := db.CreateTable([]*User{}, ifExists)
	if err != nil {
		panic(err)
	}
	go func() {
		for {
			ctx, cancel := context.WithTimeout(context.Background(), PingTimeout)
			defer cancel()
			err := db.PingContext(ctx)
			if err != nil {
				connected = false
				log.Errorf("attempting to reconnect, connection: %s", err.Error())
				db, _ = Connect("mysql", ConnectURL(devEnv))
			} else {
				connected = true
			}
			time.Sleep(SleepTimeout)
		}
	}()

}
